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

"""Tests for conv3d_temporal_unroll kernel."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.conv.conv3d_temporal_unroll import conv3d_temporal_unroll
from nkilib_src.nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_inputs(
    batch,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    filter_d,
    filter_h,
    filter_w,
    stride_d,
    stride_h,
    stride_w,
    pad_d_left,
    pad_d_right,
    pad_h_top,
    pad_h_bottom,
    pad_w_left,
    pad_w_right,
    dilation_d,
    dilation_h,
    dilation_w,
    use_bias,
    activation_fn,
    lnc_shard,
    dtype,
):
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()
    x_in = generate_tensor(name="x_in", shape=(batch, in_channels, depth, height, width), dtype=dtype)
    filters = generate_tensor(
        name="filters", shape=(filter_d, filter_h, filter_w, in_channels, out_channels), dtype=dtype
    )
    bias = generate_tensor(name="bias", shape=(out_channels,), dtype=dtype) if use_bias else None
    return {
        "x_in": x_in,
        "filters": filters,
        "bias": bias,
        "stride": (stride_d, stride_h, stride_w),
        "padding": (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right),
        "dilation": (dilation_d, dilation_h, dilation_w),
        "activation_fn": activation_fn,
        "lnc_shard": lnc_shard,
    }


PARAM_NAMES = (
    "batch, in_channels, out_channels, depth, height, width, "
    "filter_d, filter_h, filter_w, stride_d, stride_h, stride_w, "
    "pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, "
    "dilation_d, dilation_h, dilation_w, use_bias, activation_fn, lnc_shard, dtype"
)

# fmt: off
TEMPORAL_UNROLL_FAST_PARAMS = [
    # No padding
    (1, 96, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 96, 3, 3, 4, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16),
    # With padding (the target case pattern)
    (1, 96, 3, 3, 8, 8, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 96, 3, 3, 8, 8, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16),
    # Larger D (D_out=4, max for single PSUM bank with COL_TILE=32: 4*32=128=P_MAX)
    (1, 96, 3, 4, 8, 8, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16),
]

# Excluded from fast suite: compilation exceeds memory limit or takes too long
TEMPORAL_UNROLL_SLOW_PARAMS = [
    # 480p target case (Wan2.2 VAE decoder output conv)
    (1, 96, 3, 3, 480, 832, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # 720p target case (Wan2.2 VAE decoder output conv)
    (1, 96, 3, 3, 720, 1280, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
]

TEMPORAL_UNROLL_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) for c in TEMPORAL_UNROLL_FAST_PARAMS
] + TEMPORAL_UNROLL_SLOW_PARAMS
# fmt: on


@pytest_test_metadata(name="Conv3D_TemporalUnroll")
@pytest_marks(["conv", "conv3d", "temporal_unroll"])
@final
class TestConv3DTemporalUnroll:
    @pytest.mark.parametrize(PARAM_NAMES, TEMPORAL_UNROLL_ALL_PARAMS)
    def test_conv3d_temporal_unroll(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        filter_d,
        filter_h,
        filter_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d_left,
        pad_d_right,
        pad_h_top,
        pad_h_bottom,
        pad_w_left,
        pad_w_right,
        dilation_d,
        dilation_h,
        dilation_w,
        use_bias,
        activation_fn,
        lnc_shard,
        dtype,
    ) -> None:
        D_out = (depth + pad_d_left + pad_d_right - dilation_d * (filter_d - 1) - 1) // stride_d + 1
        H_out = (height + pad_h_top + pad_h_bottom - dilation_h * (filter_h - 1) - 1) // stride_h + 1
        W_out = (width + pad_w_left + pad_w_right - dilation_w * (filter_w - 1) - 1) // stride_w + 1

        def input_generator(test_config):
            return generate_inputs(
                batch,
                in_channels,
                out_channels,
                depth,
                height,
                width,
                filter_d,
                filter_h,
                filter_w,
                stride_d,
                stride_h,
                stride_w,
                pad_d_left,
                pad_d_right,
                pad_h_top,
                pad_h_bottom,
                pad_w_left,
                pad_w_right,
                dilation_d,
                dilation_h,
                dilation_w,
                use_bias,
                activation_fn,
                lnc_shard,
                dtype,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv3d_temporal_unroll,
            torch_ref=torch_ref_wrapper(conv3d_torch_ref),
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
