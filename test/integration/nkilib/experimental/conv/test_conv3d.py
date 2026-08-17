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

"""Tests for Conv3D kernel using UnitTestFramework."""

from typing import Optional, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.experimental.conv.conv3d import conv3d
from nkilib_src.nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_conv3d_inputs(
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
    use_bias: bool,
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
    dtype,
):
    """
    Generate inputs for conv3d kernel test.

    Args:
        batch: Batch size (B)
        in_channels: Number of input channels (C_in)
        out_channels: Number of output channels (C_out)
        depth: Input depth (D)
        height: Input height (H)
        width: Input width (W)
        filter_d: Filter depth (K_d)
        filter_h: Filter height (K_h)
        filter_w: Filter width (K_w)
        stride_d: Stride in depth dimension
        stride_h: Stride in height dimension
        stride_w: Stride in width dimension
        pad_d_left: Left padding in depth dimension
        pad_d_right: Right padding in depth dimension
        pad_h_top: Top padding in height dimension
        pad_h_bottom: Bottom padding in height dimension
        pad_w_left: Left padding in width dimension
        pad_w_right: Right padding in width dimension
        dilation_d: Dilation in depth dimension
        dilation_h: Dilation in height dimension
        dilation_w: Dilation in width dimension
        use_bias: Whether to include bias in convolution
        activation_fn: Activation function to apply (or None)
        lnc_shard: Whether to enable LNC sharding
        dtype: Data type for tensors

    Returns:
        dict: Dictionary of input tensors and parameters for kernel
    """
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


# fmt: off
# Parameter names for pytest.mark.parametrize
CONV3D_PARAM_NAMES = (
    "batch, in_channels, out_channels, depth, height, width, "
    "filter_d, filter_h, filter_w, stride_d, stride_h, stride_w, "
    "pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, "
    "dilation_d, dilation_h, dilation_w, use_bias, activation_fn, lnc_shard, dtype"
)

# Basic test parameters
CONV3D_BASIC_PARAMS = [
    # Small
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16),
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),

    # Medium
    (1, 48, 96, 4, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
    pytest.param(2, 96, 192, 4, 12, 12, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.SiLU, False, nl.bfloat16, marks=pytest.mark.fast),

    # Large
    (1, 128, 256, 4, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 128, 256, 4, 16, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.SiLU, False, nl.bfloat16),
    (1, 128, 512, 4, 16, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
    (1, 32, 1024, 21, 7, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 32, 1024, 21, 7, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.bfloat16),

    # Kernel size
    pytest.param(1, 32, 64, 4, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, marks=pytest.mark.fast),
    (1, 32, 64, 4, 8, 16, 3, 2, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
    (1, 32, 64, 8, 8, 16, 5, 3, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16),

    # Stride
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    pytest.param(1, 64, 128, 8, 16, 16, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, marks=pytest.mark.fast),

    # Dilation
    (1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, False, None, False, nl.bfloat16),
    (1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, True, ActFnType.GELU, False, nl.bfloat16),
    (1, 64, 128, 8, 16, 16, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, False, ActFnType.SiLU, False, nl.bfloat16),

    # Padding
    (1, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 2, 0, 2, 0, 2, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 64, 128, 4, 12, 12, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
    (1, 128, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.SiLU, False, nl.bfloat16),

    # Batch size
    (1, 64, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (2, 64, 128, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (4, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.SiLU, False, nl.bfloat16),

    # LNC sharding
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.bfloat16),
    (1, 128, 256, 4, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, True, nl.bfloat16),
    (2, 96, 192, 4, 12, 12, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.SiLU, True, nl.bfloat16),

    # float32
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.SiLU, False, nl.float32),
    pytest.param(1, 128, 512, 4, 16, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.float32, marks=pytest.mark.fast),

    # Activations
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, marks=pytest.mark.fast),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
    (1, 64, 128, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16),

    # Stride and dilation on all 3 dimensions
    (1, 128, 128, 16, 32, 64, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, False, None, False, nl.bfloat16),

    # Large spatial with DH group crossing depth boundary (regression test for NKILIB-1424).
    (1, 128, 128, 8, 42, 10, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 256, 256, 8, 80, 10, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16),

]
CONV3D_BASIC_SLOW_PARAMS = [
    (1, 128, 128, 64, 128, 256, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, True, ActFnType.GELU, False, nl.bfloat16),
]

CONV3D_BASIC_ALL_PARAMS = CONV3D_BASIC_PARAMS + CONV3D_BASIC_SLOW_PARAMS

# Wan2.2 test parameters
CONV3D_WAN2_2_VAE_ENCODER_PARAMS = [
    # Downsample[1] ResBlock first: CausalConv3d(160, 320, 3, padding=1), T=4, H=120, W=208
    # Downsample[2] time_conv: CausalConv3d(640, 640, (3,1,1), stride=(2,1,1)), T=3, H=30, W=52
    (2, 640, 640, 3, 30, 52, 3, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16),
    # Downsample[3]/Middle ResBlock: CausalConv3d(640, 640, 3, padding=1), T=1, H=30, W=52
    pytest.param(2, 640, 640, 1, 30, 52, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16, marks=pytest.mark.fast),
]

CONV3D_WAN2_2_VAE_DECODER_PARAMS = [
    # Middle ResBlock: CausalConv3d(1024, 1024, 3, padding=1), T=1, H=30, W=52
    (2, 1024, 1024, 1, 30, 52, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Upsample[1] ResBlock: CausalConv3d(1024, 1024, 3, padding=1), T=2, H=60, W=104
]

CONV3D_WAN2_2_VAE_DECODER_PARAMS2 = [
    #B, Ci, Co,  D, H, W,  Kd,Kh,Kw,  Sd,Sh,Sw, PadDL/R,HT/B,WL/R, DilD,H,W, bias, act, lnc_shard,
    # 720p
    (1, 16, 16, 3, 90, 160, 1, 1, 1,  1, 1, 1,  0, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 16, 384, 3, 90, 160, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    #(1, 96, 96, 3, 720, 1280, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    #(1, 96, 96, 3, 720, 1280, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    #(1, 96, 3, 3, 720, 1280, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    # 480p
    pytest.param(1, 16, 16, 3, 60, 104, 1, 1, 1,   1, 1, 1,  0, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16, marks=pytest.mark.fast),
    pytest.param(1, 16, 384, 3, 60, 104, 3, 3, 3,   1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16, marks=pytest.mark.fast),
    (1, 384, 384, 3, 60, 104, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 384, 768, 3, 120, 208, 3, 1, 1,  1, 1, 1,  2, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 192, 384, 3, 120, 208, 3, 1, 1,  1, 1, 1,  2, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16),
    pytest.param(1, 192, 96, 3, 240, 416, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16, marks=pytest.mark.fast),
]

# Slow-to-compile Wan2.2 VAE decoder cases (>5min compile), excluded from fast suite
CONV3D_WAN2_2_SLOW_PARAMS = [
    # 480p decoder (>2min compile time)
    (1, 96, 96, 3, 480, 832, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 96, 96, 3, 480, 832, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 96, 3, 3, 480, 832, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    # Encoder/decoder large channel configs (>40s compile)
    (2, 320, 640, 2, 60, 104, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    (2, 1024, 2048, 2, 60, 104, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16),
    (2, 1024, 1024, 1, 30, 104, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # 720p decoder large spatial dims (>2min compile time)
    (1, 384, 384, 3, 180, 320, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 384, 192, 3, 180, 320, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 192, 384, 3, 180, 320, 3, 1, 1,  1, 1, 1,  2, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 192, 96, 3, 360, 640, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    # Downsample[1] ResBlock: CausalConv3d(320, 320, 3, padding=1), T=4, H=120, W=208
    (1, 320, 320, 4, 120, 208, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Downsample[1] ResBlock first: CausalConv3d(160, 320, 3, padding=1), T=4, H=120, W=208
    (1, 160, 320, 4, 120, 208, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Downsample[2] ResBlock: CausalConv3d(640, 640, 3, padding=1), T=2, H=60, W=104
    (2, 640, 640, 2, 60, 104, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # 720p decoder
    (1, 384, 768, 3, 180, 320, 3, 1, 1,  1, 1, 1,  2, 0, 0, 0, 0, 0,  1, 1, 1, True, None, True, nl.bfloat16),
    # 480p decoder
    (1, 384, 192, 3, 120, 208, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    # Input conv: Conv3d(3, 160, 3, padding=1), T=4, H=240, W=416
    (1, 3, 160, 4, 240, 416, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, False, None, True, nl.bfloat16),
    # 720p/480p large channel configs
    (1, 384, 384, 3, 90, 160, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 384, 384, 3, 120, 208, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    # Upsample[3] ResBlock first: CausalConv3d(512, 256, 3, padding=1), T=4, H=240, W=416
    (1, 512, 256, 4, 240, 416, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Upsample[2] ResBlock: CausalConv3d(512, 512, 3, padding=1), T=4, H=120, W=208
    (1, 512, 512, 4, 120, 208, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Downsample[0] ResBlock: CausalConv3d(160, 160, 3, padding=1), T=4, H=240, W=416
    (1, 160, 160, 4, 240, 416, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Upsample[1] ResBlock: CausalConv3d(1024, 1024, 3, padding=1), T=2, H=60, W=104
    (2, 1024, 1024, 2, 60, 104, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Upsample[3] ResBlock: CausalConv3d(256, 256, 3, padding=1), T=4, H=240, W=416
    (1, 256, 256, 4, 240, 416, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16),
    # Output conv: Conv3d(256, 3, 3, padding=1)
    (1, 256, 3, 4, 240, 416, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, False, None, True, nl.bfloat16),
    # Moved here the slow ones from CONV3D_WAN2_2_VAE_DECODER_PARAMS2
    (1, 96, 96, 3, 720, 1280, 1, 3, 3,  1, 1, 1,  0, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 96, 96, 3, 720, 1280, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
    (1, 96, 3, 3, 720, 1280, 3, 3, 3,  1, 1, 1,  2, 0, 1, 1, 1, 1,  1, 1, 1, True, None, True, nl.bfloat16),
]


CONV3D_WAN2_2_FAST_PARAMS = (
    CONV3D_WAN2_2_VAE_ENCODER_PARAMS
    + CONV3D_WAN2_2_VAE_DECODER_PARAMS
    + CONV3D_WAN2_2_VAE_DECODER_PARAMS2
)

# Whisper model test parameters
CONV3D_WHISPER_PARAMS = [
    # Conv1
    (2, 80, 384, 1, 1, 1500, 1, 1, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.float32),
    pytest.param(1, 128, 1280, 1, 1, 1500, 1, 1, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, True, nl.bfloat16, marks=pytest.mark.fast),

    # Conv2
    (2, 384, 384, 1, 1, 1500, 1, 1, 3, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.float32),
    (1, 1280, 1280, 1, 1, 1500, 1, 1, 3, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, True, nl.bfloat16),
]

# Conv2D test parameters
CONV3D_CONV2D_STYLE_PARAMS = [
    # Conv2D basic
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 64, 128, 1, 32, 32, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),

    # Conv2D with stride= on H and W
    (1, 64, 128, 1, 16, 16, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 128, 256, 1, 32, 32, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16),

    # Conv2D with stride and dilation on H and W
    pytest.param(1, 32, 64, 1, 32, 32, 1, 3, 3, 1, 2, 2, 0, 0, 2, 2, 2, 2, 1, 2, 2, False, None, False, nl.bfloat16, marks=pytest.mark.fast),
    (1, 64, 128, 1, 32, 32, 1, 3, 3, 1, 2, 2, 0, 0, 2, 2, 2, 2, 1, 2, 2, True, ActFnType.GELU, False, nl.bfloat16),

    # Conv2D large
    (1, 128, 256, 1, 56, 56, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16),
    (1, 256, 512, 1, 28, 28, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16),
]
# fmt: on


@pytest_test_metadata(name="Conv3D")
@pytest_marks(["conv", "conv3d"])
@final
class TestConv3DKernel:
    """Test class for Conv3D kernel validation using UnitTestFramework."""

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_BASIC_ALL_PARAMS)
    def test_conv3d_basic(
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
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run basic Conv3D tests covering various convolution configurations."""
        self._run_conv3d_test(
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
            pad_d_left=pad_d_left,
            pad_d_right=pad_d_right,
            pad_h_top=pad_h_top,
            pad_h_bottom=pad_h_bottom,
            pad_w_left=pad_w_left,
            pad_w_right=pad_w_right,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_WAN2_2_FAST_PARAMS)
    def test_conv3d_wan2_2(
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
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run Wan2.2 model Conv3D tests."""
        self._run_conv3d_test(
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
            pad_d_left=pad_d_left,
            pad_d_right=pad_d_right,
            pad_h_top=pad_h_top,
            pad_h_bottom=pad_h_bottom,
            pad_w_left=pad_w_left,
            pad_w_right=pad_w_right,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_WHISPER_PARAMS)
    def test_conv3d_whisper(
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
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run Whisper model Conv3D tests for Conv1 and Conv2 layer configurations."""
        self._run_conv3d_test(
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
            pad_d_left=pad_d_left,
            pad_d_right=pad_d_right,
            pad_h_top=pad_h_top,
            pad_h_bottom=pad_h_bottom,
            pad_w_left=pad_w_left,
            pad_w_right=pad_w_right,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_CONV2D_STYLE_PARAMS)
    def test_conv3d_conv2d_style(
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
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run Conv2D tests using Conv3D with D=1, K_d=1."""
        self._run_conv3d_test(
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
            pad_d_left=pad_d_left,
            pad_d_right=pad_d_right,
            pad_h_top=pad_h_top,
            pad_h_bottom=pad_h_bottom,
            pad_w_left=pad_w_left,
            pad_w_right=pad_w_right,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    def _run_conv3d_test(
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
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """
        Execute a Conv3D kernel test with specified parameters using UnitTestFramework.

        Args:
            test_manager: Test orchestrator for kernel execution
            batch: Batch size (B)
            in_channels: Number of input channels (C_in)
            out_channels: Number of output channels (C_out)
            depth: Input depth (D)
            height: Input height (H)
            width: Input width (W)
            filter_d: Filter depth (K_d)
            filter_h: Filter height (K_h)
            filter_w: Filter width (K_w)
            stride_d: Stride in depth dimension
            stride_h: Stride in height dimension
            stride_w: Stride in width dimension
            pad_d_left: Left padding in depth dimension
            pad_d_right: Right padding in depth dimension
            pad_h_top: Top padding in height dimension
            pad_h_bottom: Bottom padding in height dimension
            pad_w_left: Left padding in width dimension
            pad_w_right: Right padding in width dimension
            dilation_d: Dilation in depth dimension
            dilation_h: Dilation in height dimension
            dilation_w: Dilation in width dimension
            use_bias: Whether to include bias in convolution
            activation_fn: Activation function to apply (or None)
            lnc_shard: Whether to enable LNC sharding
            dtype: Data type for tensors
        """
        # Calculate output shape
        D_out = (depth + pad_d_left + pad_d_right - dilation_d * (filter_d - 1) - 1) // stride_d + 1
        H_out = (height + pad_h_top + pad_h_bottom - dilation_h * (filter_h - 1) - 1) // stride_h + 1
        W_out = (width + pad_w_left + pad_w_right - dilation_w * (filter_w - 1) - 1) // stride_w + 1

        def input_generator(test_config):
            return generate_conv3d_inputs(
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
                use_bias=use_bias,
                activation_fn=activation_fn,
                lnc_shard=lnc_shard,
                dtype=dtype,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv3d,
            torch_ref=torch_ref_wrapper(conv3d_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        # Use higher tolerance for bfloat16
        rtol = 1e-2 if dtype == nl.bfloat16 else 1e-5
        atol = 1e-2 if dtype == nl.bfloat16 else 1e-5

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )
