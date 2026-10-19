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

"""Tests for the Conv3D backward kernel (dx, dw, db) using UnitTestFramework."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.conv.conv3d_bwd import conv3d_bwd
from nkilib_src.nkilib.experimental.conv.conv3d_bwd_torch import conv3d_bwd_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _out_dim(in_size, k, stride, pad_lo, pad_hi, dil):
    return (in_size + pad_lo + pad_hi - dil * (k - 1) - 1) // stride + 1


def generate_conv3d_bwd_inputs(
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
    pad_d_left: int,
    pad_d_right: int,
    pad_h_top: int,
    pad_h_bottom: int,
    pad_w_left: int,
    pad_w_right: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    compute_dx: bool,
    compute_dw: bool,
    compute_db: bool,
    dtype,
):
    """
    Generate kernel inputs for a conv3d_bwd call.

    Produces a forward input x_in [B, C_in, D, H, W], forward filters
    [K_d, K_h, K_w, C_in, C_out], and an output gradient dy
    [B, C_out, D_out, H_out, W_out] with 6-value asymmetric padding.
    """
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()

    D_out = _out_dim(depth, filter_d, stride_d, pad_d_left, pad_d_right, dilation_d)
    H_out = _out_dim(height, filter_h, stride_h, pad_h_top, pad_h_bottom, dilation_h)
    W_out = _out_dim(width, filter_w, stride_w, pad_w_left, pad_w_right, dilation_w)

    x_in = generate_tensor(name="x_in", shape=(batch, in_channels, depth, height, width), dtype=dtype)
    filters = generate_tensor(
        name="filters",
        shape=(filter_d, filter_h, filter_w, in_channels, out_channels),
        dtype=dtype,
    )
    dy = generate_tensor(name="dy", shape=(batch, out_channels, D_out, H_out, W_out), dtype=dtype)

    return {
        "dy": dy,
        "x_in": x_in,
        "filters": filters,
        "stride": (stride_d, stride_h, stride_w),
        "padding": (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right),
        "dilation": (dilation_d, dilation_h, dilation_w),
        "compute_dx": compute_dx,
        "compute_dw": compute_dw,
        "compute_db": compute_db,
    }


# fmt: off
CONV3D_BWD_PARAM_NAMES = (
    "batch, in_channels, out_channels, depth, height, width, "
    "filter_d, filter_h, filter_w, stride_d, stride_h, stride_w, "
    "pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, "
    "dilation_d, dilation_h, dilation_w, compute_dx, compute_dw, compute_db, dtype"
)

# Basic test parameters
CONV3D_BWD_BASIC_PARAMS = [
    # Small
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),

    # Medium
    (1, 48, 96, 4, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    pytest.param(2, 96, 192, 4, 12, 12, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16, marks=pytest.mark.fast),

    # Large
    (1, 128, 256, 4, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 128, 256, 4, 16, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 32, 1024, 21, 7, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),

    # Kernel size
    pytest.param(1, 32, 64, 4, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16, marks=pytest.mark.fast),
    (1, 32, 64, 4, 8, 16, 3, 2, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 32, 64, 8, 8, 16, 5, 3, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),

    # Stride
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    pytest.param(1, 64, 128, 8, 16, 16, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16, marks=pytest.mark.fast),

    # Dilation
    (1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, True, True, True, nl.bfloat16),
    (1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, True, True, True, nl.bfloat16),
    (1, 64, 128, 8, 16, 16, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, True, True, True, nl.bfloat16),

    # Padding (asymmetric)
    (1, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 2, 0, 2, 0, 2, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 64, 128, 4, 12, 12, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 128, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),

    # Batch size
    (1, 64, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (2, 64, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (4, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),

    # LNC sharding
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 128, 256, 4, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),
    (2, 96, 192, 4, 12, 12, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),

    # float32
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.float32, marks=pytest.mark.fast),
    pytest.param(1, 128, 256, 4, 16, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.float32),

    # Stride and dilation on all 3 dimensions
    (1, 128, 128, 16, 32, 64, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, True, True, True, nl.bfloat16),

    # Large spatial with DH group crossing depth boundary
    (1, 128, 128, 8, 42, 10, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),
    (1, 256, 256, 8, 80, 10, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),

    # Subset selection (gradient masking)
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, True, False, nl.bfloat16, marks=pytest.mark.fast),
    pytest.param(2, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, False, True, nl.bfloat16, marks=pytest.mark.fast),

    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, False, nl.bfloat16, marks=pytest.mark.slow),
]

# Large vision benchmark workloads
CONV3D_LARGE_VISION_PARAMS = [
    pytest.param(6, 32, 32, 1, 432, 768, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, False, nl.bfloat16, marks=pytest.mark.slow),
    pytest.param(6, 64, 64, 1, 216, 384, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, False, nl.bfloat16, marks=pytest.mark.slow),
    pytest.param(6, 128, 128, 1, 108, 192, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, False, nl.bfloat16, marks=pytest.mark.slow),
    pytest.param(6, 256, 256, 1, 54, 96, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, False, nl.bfloat16, marks=pytest.mark.slow),
]

# ResNet-50 backward configs
CONV3D_BWD_RESNET50_PARAMS = [
    # B, C_in, C_out, D,   H,   W, K_d, K_h, K_w, s_d, s_h, s_w, pd_lo, pd_hi, ph_lo, ph_hi, pw_lo, pw_hi, dil_d, dil_h, dil_w, dx, dw, db, dtype
    (2, 3, 64, 1, 224, 224, 1, 7, 7, 1, 2, 2, 0, 0, 3, 3, 3, 3, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 0] 1x conv1 (stem 7x7) — B=4, target 267us
    (2, 64, 64, 1, 56, 56, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 1] 1x layer1.0.conv1
    (2, 64, 64, 1, 56, 56, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 2] 3x layer1.*.conv2
    (2, 64, 256, 1, 56, 56, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 3] 4x layer1.*.conv3
    (2, 256, 64, 1, 56, 56, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 4] 2x layer1.1+.conv1
    (2, 256, 128, 1, 56, 56, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 5] 1x layer2.0    .conv1
    (2, 128, 128, 1, 56, 56, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 6] 1x layer2.0.conv2 (stride2)
    (2, 128, 512, 1, 28, 28, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 7] 4x layer2.*.conv3
    (2, 256, 512, 1, 56, 56, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 8] 1x layer2.0.downsample.0
    (2, 512, 128, 1, 28, 28, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [ 9] 3x layer2.1+.conv1
    (2, 128, 128, 1, 28, 28, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [10] 3x layer2.1+.conv2
    (2, 512, 256, 1, 28, 28, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [11] 1x layer3.0.conv1
    (2, 256, 256, 1, 28, 28, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [12] 1x layer3.0.conv2 (stride2)
    (2, 256, 1024, 1, 14, 14, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [13] 6x layer3.*.conv3
    (2, 512, 1024, 1, 28, 28, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [14] 1x layer3.0.downsample.0
    (2, 1024, 256, 1, 14, 14, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [15] 5x layer3.1+.conv1
    (2, 256, 256, 1, 14, 14, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [16] 5x layer3.1+.conv2
    (2, 1024, 512, 1, 14, 14, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [17] 1x layer4.0.conv1
    (2, 512, 512, 1, 14, 14, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [18] 1x layer4.0.conv2 (stride2)
    (2, 512, 2048, 1, 7, 7, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [19] 3x layer4.*.conv3
    (2, 1024, 2048, 1, 14, 14, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [20] 1x layer4.0.downsample.0
    (2, 2048, 512, 1, 7, 7, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, True, True, nl.bfloat16),  # [21] 2x layer4.1+.conv1
    (2, 512, 512, 1, 7, 7, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, True, True, nl.bfloat16),  # [22] 2x layer4.1+.conv2
]
# fmt: on


@pytest_test_metadata(name="Conv3DBwd")
@pytest_marks(["conv", "conv3d", "conv3d_bwd"])
@final
class TestConv3DBwdKernel:
    """Test class for the Conv3D backward kernel validation using UnitTestFramework."""

    def _run_conv3d_bwd(
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
        pad_d_left: int,
        pad_d_right: int,
        pad_h_top: int,
        pad_h_bottom: int,
        pad_w_left: int,
        pad_w_right: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        compute_dx: bool,
        compute_dw: bool,
        compute_db: bool,
        dtype,
    ) -> None:
        """Run a conv3d_bwd kernel test against a torch.autograd reference."""

        def input_generator(test_config):
            return generate_conv3d_bwd_inputs(
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
                pad_d_left=pad_d_left,
                pad_d_right=pad_d_right,
                pad_h_top=pad_h_top,
                pad_h_bottom=pad_h_bottom,
                pad_w_left=pad_w_left,
                pad_w_right=pad_w_right,
                dilation_d=dilation_d,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                compute_dx=compute_dx,
                compute_dw=compute_dw,
                compute_db=compute_db,
                dtype=dtype,
            )

        def output_tensors(kernel_input):
            outputs = {}
            if compute_dx:
                outputs["dx"] = np.zeros((batch, in_channels, depth, height, width), dtype=dtype)
            if compute_dw:
                outputs["dw"] = np.zeros((filter_d, filter_h, filter_w, in_channels, out_channels), dtype=dtype)
            if compute_db:
                outputs["db"] = np.zeros((out_channels,), dtype=dtype)
            return outputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv3d_bwd,
            torch_ref=torch_ref_wrapper(conv3d_bwd_torch_ref),
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

    @pytest.mark.parametrize(CONV3D_BWD_PARAM_NAMES, CONV3D_BWD_BASIC_PARAMS)
    def test_conv3d_bwd_basic(
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
        pad_d_left: int,
        pad_d_right: int,
        pad_h_top: int,
        pad_h_bottom: int,
        pad_w_left: int,
        pad_w_right: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        compute_dx: bool,
        compute_dw: bool,
        compute_db: bool,
        dtype,
    ) -> None:
        """Run conv3d_bwd over the basic tests."""
        self._run_conv3d_bwd(
            test_manager,
            platform_target,
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
            compute_dx,
            compute_dw,
            compute_db,
            dtype,
        )

    @pytest.mark.parametrize(CONV3D_BWD_PARAM_NAMES, CONV3D_LARGE_VISION_PARAMS)
    def test_conv3d_bwd_large_vision(
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
        pad_d_left: int,
        pad_d_right: int,
        pad_h_top: int,
        pad_h_bottom: int,
        pad_w_left: int,
        pad_w_right: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        compute_dx: bool,
        compute_dw: bool,
        compute_db: bool,
        dtype,
    ) -> None:
        """Run conv3d_bwd over the large vision benchmark workloads."""
        self._run_conv3d_bwd(
            test_manager,
            platform_target,
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
            compute_dx,
            compute_dw,
            compute_db,
            dtype,
        )

    @pytest.mark.parametrize(CONV3D_BWD_PARAM_NAMES, CONV3D_BWD_RESNET50_PARAMS)
    def test_conv3d_bwd_resnet50(
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
        pad_d_left: int,
        pad_d_right: int,
        pad_h_top: int,
        pad_h_bottom: int,
        pad_w_left: int,
        pad_w_right: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        compute_dx: bool,
        compute_dw: bool,
        compute_db: bool,
        dtype,
    ) -> None:
        """Run conv3d_bwd over every distinct ResNet-50 conv config."""
        self._run_conv3d_bwd(
            test_manager,
            platform_target,
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
            compute_dx,
            compute_dw,
            compute_db,
            dtype,
        )
