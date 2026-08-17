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

"""Tests for Conv3DTranspose kernel using UnitTestFramework."""

from typing import Optional, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.experimental.conv.conv3d_transpose import conv3d_transpose
from nkilib_src.nkilib.experimental.conv.conv3d_transpose_torch import conv3d_transpose_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_conv3d_transpose_inputs(
    batch: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    filter_d: int,
    filter_h: int,
    filter_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    use_bias: bool,
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
    filter_shape: str,
    dtype,
    output_pad_d: int = 0,
    output_pad_h: int = 0,
    output_pad_w: int = 0,
    flip_filter_taps_in_kernel: bool = False,
):
    """
    Generate kernel inputs for a Conv3DTranspose call.

    Builds the filter in a forward-conv layout with (C_in=in_channels,
    C_out=out_channels). When flip_filter_taps_in_kernel is True the kernel reads spatially-reversed taps internally.
    If filter_shape == "KDHW_CO_CI", the channel axes are also swapped so
    position 3 holds C_out and position 4 holds C_in.
    """
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()

    x_in = generate_tensor(name="x_in", shape=(batch, in_channels, depth, height, width), dtype=dtype)

    filter_ci_co = generate_tensor(
        name="filter_ci_co",
        shape=(filter_d, filter_h, filter_w, in_channels, out_channels),
        dtype=dtype,
    )
    if flip_filter_taps_in_kernel:
        filter_ci_co_oriented = filter_ci_co
    else:
        filter_ci_co_oriented = filter_ci_co[::-1, ::-1, ::-1, :, :].copy()

    if filter_shape == "KDHW_CI_CO":
        kernel_filter = np.ascontiguousarray(filter_ci_co_oriented)
    elif filter_shape == "KDHW_CO_CI":
        kernel_filter = np.ascontiguousarray(np.transpose(filter_ci_co_oriented, (0, 1, 2, 4, 3)))
    else:
        raise ValueError(f"Unsupported filter_shape '{filter_shape}'")

    bias = generate_tensor(name="bias", shape=(out_channels,), dtype=dtype) if use_bias else None

    return {
        "x_in": x_in,
        "filters": kernel_filter,
        "bias": bias,
        "stride": (stride_d, stride_h, stride_w),
        "padding": (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w),
        "dilation": (dilation_d, dilation_h, dilation_w),
        "output_padding": (output_pad_d, output_pad_h, output_pad_w),
        "activation_fn": activation_fn,
        "lnc_shard": lnc_shard,
        "filter_shape": filter_shape,
        "flip_filter_taps_in_kernel": flip_filter_taps_in_kernel,
    }


# fmt: off
CONV3D_TRANSPOSE_PARAM_NAMES = (
    "batch, in_channels, out_channels, depth, height, width, "
    "filter_d, filter_h, filter_w, "
    "stride_d, stride_h, stride_w, "
    "pad_d, pad_h, pad_w, "
    "dilation_d, dilation_h, dilation_w, "
    "use_bias, activation_fn, lnc_shard, filter_shape, dtype, "
    "output_pad_d, output_pad_h, output_pad_w, flip_filter_taps_in_kernel"
)

# Basic test parameters
CONV3D_TRANSPOSE_BASIC_PARAMS_LIST = [
    (1, 256, 256, 1, 100,  58, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    (6, 256, 256, 1, 100,  58, 1, 3, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    (6, 512, 512, 1,  40,  23, 1, 3, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 2, 0, 0, 1, 1, 1, 1, False, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, marks=pytest.mark.fast),
    pytest.param(1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, marks=pytest.mark.fast),
    # Slow test parameters (long compilation time >10 minutes).
    (6, 512, 512, 1,  50,  29, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    (6, 256, 256, 1,  80,  46, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    (6, 512, 512, 1,  40,  23, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
]

# Conv3D bwd usage test parameters
CONV3D_TRANSPOSE_BWD_PARAMS_LIST = [
    (1, 256, 256, 1, 100, 58, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, False),
    pytest.param(1, 16, 32, 1, 8, 8, 1, 3, 3, 1, 3, 1, 0, 0, 0, 1, 1, 1, False, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 1, 0, False, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 1, 8, 8, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CO_CI", nl.bfloat16, 0, 0, 0, True, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 1, 8, 8, 1, 3, 3, 1, 3, 1, 0, 0, 0, 1, 1, 1, False, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 1, 0, True, marks=pytest.mark.fast),
]
# fmt: on


@pytest_test_metadata(name="Conv3DTranspose")
@pytest_marks(["conv", "conv3d", "conv3d_transpose"])
@final
class TestConv3DTransposeKernel:
    """Test class for Conv3DTranspose kernel validation using UnitTestFramework."""

    def _run_conv3d_transpose(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        in_channels: int,
        out_channels: int,
        depth: int,
        height: int,
        width: int,
        filter_d: int,
        filter_h: int,
        filter_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        filter_shape: str,
        dtype,
        output_pad_d: int,
        output_pad_h: int,
        output_pad_w: int,
        flip_filter_taps_in_kernel: bool,
    ) -> None:
        """Run a Conv3DTranspose kernel test against a torch ConvTranspose3d reference."""
        D_out = (depth - 1) * stride_d + dilation_d * (filter_d - 1) - 2 * pad_d + output_pad_d + 1
        H_out = (height - 1) * stride_h + dilation_h * (filter_h - 1) - 2 * pad_h + output_pad_h + 1
        W_out = (width - 1) * stride_w + dilation_w * (filter_w - 1) - 2 * pad_w + output_pad_w + 1

        def input_generator(test_config):
            return generate_conv3d_transpose_inputs(
                batch=batch,
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                height=height,
                width=width,
                filter_d=filter_d,
                filter_h=filter_h,
                filter_w=filter_w,
                stride_d=stride_d,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_d=pad_d,
                pad_h=pad_h,
                pad_w=pad_w,
                dilation_d=dilation_d,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                use_bias=use_bias,
                activation_fn=activation_fn,
                lnc_shard=lnc_shard,
                filter_shape=filter_shape,
                dtype=dtype,
                output_pad_d=output_pad_d,
                output_pad_h=output_pad_h,
                output_pad_w=output_pad_w,
                flip_filter_taps_in_kernel=flip_filter_taps_in_kernel,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv3d_transpose,
            torch_ref=torch_ref_wrapper(conv3d_transpose_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        rtol = 1e-2 if dtype == nl.bfloat16 else 1e-5
        atol = 1e-2 if dtype == nl.bfloat16 else 1e-5

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize(CONV3D_TRANSPOSE_PARAM_NAMES, CONV3D_TRANSPOSE_BASIC_PARAMS_LIST)
    def test_conv3d_transpose_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        in_channels: int,
        out_channels: int,
        depth: int,
        height: int,
        width: int,
        filter_d: int,
        filter_h: int,
        filter_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        filter_shape: str,
        dtype,
        output_pad_d: int,
        output_pad_h: int,
        output_pad_w: int,
        flip_filter_taps_in_kernel: bool,
    ) -> None:
        """Run basic Conv3DTranspose kernel tests."""
        self._run_conv3d_transpose(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            height=height,
            width=width,
            filter_d=filter_d,
            filter_h=filter_h,
            filter_w=filter_w,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            filter_shape=filter_shape,
            dtype=dtype,
            output_pad_d=output_pad_d,
            output_pad_h=output_pad_h,
            output_pad_w=output_pad_w,
            flip_filter_taps_in_kernel=flip_filter_taps_in_kernel,
        )

    @pytest.mark.parametrize(CONV3D_TRANSPOSE_PARAM_NAMES, CONV3D_TRANSPOSE_BWD_PARAMS_LIST)
    def test_conv3d_transpose_bwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        in_channels: int,
        out_channels: int,
        depth: int,
        height: int,
        width: int,
        filter_d: int,
        filter_h: int,
        filter_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        filter_shape: str,
        dtype,
        output_pad_d: int,
        output_pad_h: int,
        output_pad_w: int,
        flip_filter_taps_in_kernel: bool,
    ) -> None:
        """Exercise output_padding and flip_filter_taps_in_kernel."""
        self._run_conv3d_transpose(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            height=height,
            width=width,
            filter_d=filter_d,
            filter_h=filter_h,
            filter_w=filter_w,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            filter_shape=filter_shape,
            dtype=dtype,
            output_pad_d=output_pad_d,
            output_pad_h=output_pad_h,
            output_pad_w=output_pad_w,
            flip_filter_taps_in_kernel=flip_filter_taps_in_kernel,
        )
