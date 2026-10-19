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
from nkilib_src.nkilib.experimental.conv.conv3d import BatchNormMode, ResidualAddLoc, conv3d
from nkilib_src.nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref
from nkilib_src.nkilib.experimental.conv.conv_fuser_torch import conv_fuser_torch_ref

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
    batch_norm_mode: BatchNormMode = BatchNormMode.NONE,
    output_pre_norm: bool = False,
    batch_norm_eps: float = 1e-5,
    momentum: float = 0.1,
    residual_add_loc: ResidualAddLoc = ResidualAddLoc.NONE,
    omit_residuals_in: bool = False,
    x_in_mean: float = 0.0,
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
        residual_add_loc: Where the fused residual add goes relative to the activation
        omit_residuals_in: Skip generating residuals_in even though residual_add_loc requires it,
            to exercise the kernel's input validation
        x_in_mean: Mean of the generated x_in (default 0.0). A non-zero mean makes the conv output's
            per-channel mean large relative to its std, exercising batchnorm-rescale precision

    Returns:
        dict: Dictionary of input tensors and parameters for kernel
    """
    generate_tensor = gaussian_tensor_generator()
    # x_in_mean shifts the input off zero so the output's per-channel mean is large relative to its std;
    # that ratio amplifies any precision loss in the rescale, which zero-mean inputs hide entirely.

    # seed differs from generate_tensor's default 0: sharing it would draw x_in from the same stream as
    # filters, making (x_in - x_in_mean) the filter prefix (Pearson r = 1.0) — a degenerate conv.
    shifted_tensor = gaussian_tensor_generator(mean=x_in_mean, seed=1) if x_in_mean != 0.0 else generate_tensor

    x_in = shifted_tensor(name="x_in", shape=(batch, in_channels, depth, height, width), dtype=dtype)
    filters = generate_tensor(
        name="filters", shape=(filter_d, filter_h, filter_w, in_channels, out_channels), dtype=dtype
    )
    bias = generate_tensor(name="bias", shape=(out_channels,), dtype=dtype) if use_bias else None

    inputs = {
        "x_in": x_in,
        "filters": filters,
        "bias": bias,
        "stride": (stride_d, stride_h, stride_w),
        "padding": (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right),
        "dilation": (dilation_d, dilation_h, dilation_w),
        "activation_fn": activation_fn,
        "lnc_shard": lnc_shard,
        "batch_norm_mode": batch_norm_mode,
        # Always passed so the flag is exercised as a don't-care in BatchNormMode.NONE.
        "output_pre_norm": output_pre_norm,
        "residual_add_loc": residual_add_loc,
    }

    if residual_add_loc != ResidualAddLoc.NONE and not omit_residuals_in:
        # residuals_in matches the output shape [B, C_out, D_out, H_out, W_out] and x_in's dtype.
        D_out = (depth + pad_d_left + pad_d_right - dilation_d * (filter_d - 1) - 1) // stride_d + 1
        H_out = (height + pad_h_top + pad_h_bottom - dilation_h * (filter_h - 1) - 1) // stride_h + 1
        W_out = (width + pad_w_left + pad_w_right - dilation_w * (filter_w - 1) - 1) // stride_w + 1
        inputs["residuals_in"] = generate_tensor(
            name="residuals_in", shape=(batch, out_channels, D_out, H_out, W_out), dtype=dtype
        )

    if batch_norm_mode.is_fused():
        # gamma / beta are per-C_out-channel affine params of shape [C_out, 1], always float32.
        inputs["batch_norm_eps"] = batch_norm_eps
        inputs["gamma"] = generate_tensor(name="gamma", shape=(out_channels, 1), dtype=np.float32)
        inputs["beta"] = generate_tensor(name="beta", shape=(out_channels, 1), dtype=np.float32)
        # Running statistics carried across steps, [C_out, 1] float32. running_variances
        # must be non-negative; use the squared generator output as a simple positive init.
        inputs["momentum"] = momentum
        inputs["running_means"] = generate_tensor(name="running_means", shape=(out_channels, 1), dtype=np.float32)
        running_var = generate_tensor(name="running_variances", shape=(out_channels, 1), dtype=np.float32)
        inputs["running_variances"] = np.abs(running_var)

    return inputs


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

# Batchnorm-fusion parameters (see CONV3D_BATCH_NORM_PARAM_NAMES for the appended columns). gamma /
# beta are drawn per C_out, so distinct C_out configs also vary them. Non-obvious coverage axes:
#  * lnc_shard=True reaches the LNC=2 paths (cross-core reduction or per-core C_out slice) and the
#    B==1 / B>1 rescale split.
#  * Small B*D_out*H_out*W_out keeps the correction-1 factor N/(N-1) far from 1, verifying it.
#  * EVAL validates no statistics outputs; output_pre_norm is TRAINING-only (EVAL rejects it).
#  * With no activation the two residual_add_loc values are equivalent.
CONV3D_BATCH_NORM_PARAM_NAMES = (
    CONV3D_PARAM_NAMES + ", batch_norm_eps, momentum, batch_norm_mode, output_pre_norm, residual_add_loc"
)

CONV3D_BATCH_NORM_PARAMS = [
    # Small, no padding, no bias, default eps, default momentum
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # With bias and padding (exercises padded-output fill-in in the reduction), larger eps
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16, 1e-3, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # float32, no padding, with bias, default eps
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # C_out > P_MAX (multiple C_out tiles) with stride and padding, small eps
    (1, 64, 256, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16, 1e-6, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # Conv2D-style (D=1, K_d=1) with W tiling and padding, larger eps
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16, 1e-2, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # float32, larger eps to exercise a distinct rsqrt(var+eps) path
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-1, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # C_out > P_MAX with bias, no padding, default eps (distinct gamma / beta across tiles)
    pytest.param(1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),

    # --- momentum sweep: distinct momentum values ---
    # momentum=0.5 (equal weighting of running and new stats), float32 for tight tolerance.
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-5, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # momentum=1.0 (running stats fully replaced by new stats), with bias.
    (1, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, 1e-5, 1.0, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # momentum=0.0 (running stats unchanged), float32.
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-5, 0.0, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),

    # --- low B*D_out*H_out*W_out: correction-1 factor N/(N-1) far from 1 (N=8 here, so 8/7), validated
    # through updated_running_variances — it shifts the running variance 14-33%. bfloat16 because
    # normalizing `out` over so few samples amplifies fp32 matmul rounding past the fp32 tolerance. ---
    pytest.param(1, 16, 32, 4, 4, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # N = 1*1*2*2 = 4 (D=3,K_d=3 -> D_out=1): N/(N-1) = 4/3, with bias.
    (1, 32, 64, 3, 4, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, 1e-5, 0.3, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),

    # --- lnc_shard=True + TRAINING: statistics reduced across the two cores by whichever strategy the
    # D_out*H_out vs C_out-tile balance picks — shard_on_dh=True exchanges triples via nisa.sendrecv
    # (the 8x8 single-tile cases), shard_on_dh=False gives each core a C_out slice (5x5 C_out=256). ---
    # shard_on_dh path: B=1, single C_out tile.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # shard_on_dh path: B=2 with bias (rescale sharded on batch).
    (2, 32, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16, 1e-3, 0.2, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # shard_on_dh path: B=1, float32 (tight tolerance on the cross-core reduced stats).
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.float32, 1e-5, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # C_out-shard path: C_out=256 -> 2 C_out tiles, B=1 (each core owns one tile).
    (1, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # C_out-shard path: C_out=256, B=2 with bias.
    pytest.param(2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # shard_on_dh path + low N: verifies the correction-1 update through the cross-core
    # reduction. N=1*2*2*2=8. bfloat16 for the low-N fp32-tolerance reason noted above.
    (1, 16, 32, 4, 4, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16, 1e-5, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),

    # --- activation applied after batchnorm ---
    # SiLU after batchnorm, no bias, float32.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.SiLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # GELU after batchnorm, with bias and padding.
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # ReLU after batchnorm, no bias, float32.
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # SiLU after batchnorm + lnc_shard=True (stats reduced across cores): activation on normalized out.
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.SiLU, True, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    # GELU after batchnorm, C_out > P_MAX with bias.
    (1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),

    # --- BatchNormMode.EVAL: normalize with the given running statistics on the PSUM eviction, no
    # momentum update. Covers that path with / without bias (folded into the affine) and activation,
    # single and multiple C_out tiles, column tiling, padding, both dtypes, LNC sharding. ---
    # Small, no padding, no bias, float32 (tight tolerance on the eval affine).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # With bias and padding (bias folded into neg_true_beta; padded positions get bn(bias)).
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16, 1e-3, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # bfloat16, no bias, no padding, larger eps.
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-1, 0.5, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    # ReLU after the eval-mode batchnorm, no bias, float32.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # SiLU after the eval-mode batchnorm, with bias and padding (bias fold + activation).
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    # C_out > P_MAX (multiple C_out tiles, distinct gamma / beta / running stats per tile) + bias.
    pytest.param(1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # C_out > P_MAX with GELU, stride and padding.
    (1, 64, 256, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, 1e-6, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    # Conv2D-style (D=1, K_d=1) with W tiling and padding.
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16, 1e-2, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    # lnc_shard=True (each core normalizes its own slice; no cross-core reduction is needed in
    # eval mode since no statistics are computed). shard_on_dh-style shape, B=1.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.float32, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # lnc_shard=True, C_out=256 (C_out-shard shape), B=2 with bias and ReLU.
    (2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),

    # --- output_pre_norm=True (TRAINING only): conv_out must be the un-normalized result, bias included
    # and activation NOT applied, while y_out and the statistics stay correct. Covered: with / without
    # bias and activation, padding, both dtypes, multiple C_out tiles, both LNC strategies. ---
    # Small, no padding, no bias, float32.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # With bias and padding (padded conv_out positions hold the bias, not garbage).
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, None, False, nl.bfloat16, 1e-3, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # With activation: conv_out must be pre-batchnorm AND pre-activation.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # With bias + SiLU and padding.
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE),
    # C_out > P_MAX (multiple C_out tiles) with bias, distinct momentum.
    pytest.param(1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, 1e-5, 0.5, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # lnc_shard=True, shard_on_dh path (each core rescales its own D-H slice of conv_out), B=1.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # lnc_shard=True, C_out-shard path (C_out=256 -> 2 tiles), B=2 with bias.
    (2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE),
    # Conv2D-style (D=1, K_d=1) with W tiling and padding.
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, None, False, nl.bfloat16, 1e-2, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE),

    # --- fused residual add with a batchnorm: EVAL adds on the PSUM eviction, TRAINING in the rescale
    # phase where an activation folds the add into the affine. Covered on both axes: PRE_ACT / POST_ACT,
    # with / without activation and bias, padding, multiple C_out tiles, column tiling, LNC. ---

    # bfloat16: the add is exact either way, but normalizing with the *given* running statistics scales the
    # fp32 matmul error (~3e-5 rel) by true_gamma, which reaches ~7 at these randomly drawn variances —
    # past the fp32 tolerance though values stay correct to ~1e-4. (residuals_in shifts the random stream,
    # so these draw smaller variances than the residual-free rows.)
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # EVAL, POST_ACT with ReLU, no bias: distinguishes the two locations (ReLU clamps
    # before the add in POST_ACT only).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # EVAL, PRE_ACT with bias + SiLU and padding (bias folded into the affine; padded positions
    # must pick the residual up too).
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # EVAL, POST_ACT with bias + GELU, stride and padding, C_out > P_MAX.
    (1, 64, 256, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, 1e-6, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT),
    # EVAL, no activation: PRE_ACT and POST_ACT are equivalent, so both must give the same result.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT),
    # EVAL, lnc_shard=True (each core adds the residual on its own slice), C_out-shard shape, B=2.
    (2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT),
    # EVAL, lnc_shard=True, shard_on_dh shape, B=1.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # EVAL, Conv2D-style (D=1, K_d=1) with W tiling and padding.
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-2, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT),
    # TRAINING, PRE_ACT with ReLU, no bias: rescale-phase normalize -> add -> activate (the
    # true_beta sign flip).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # TRAINING, POST_ACT with ReLU, no bias: activation(scale, bias) + tensor_tensor fold.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # TRAINING, no activation: the plain rescale tensor_scalar + one tensor_tensor add; the
    # statistics must still be the raw (residual-free) conv statistics.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT),
    # TRAINING, PRE_ACT with bias + SiLU and padding.
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT),
    # TRAINING, POST_ACT with bias + GELU, C_out > P_MAX (multiple C_out tiles).
    pytest.param(1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # TRAINING, lnc_shard=True, shard_on_dh path (each core rescales / adds on its own D-H slice).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # TRAINING, lnc_shard=True, C_out-shard path (C_out=256 -> 2 tiles), B=2 with bias.
    (2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT),
    # TRAINING + output_pre_norm=True with a residual: conv_out must remain the raw conv output,
    # i.e. pre-batchnorm, pre-activation AND pre-residual.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-3, 0.2, BatchNormMode.TRAINING, True, ResidualAddLoc.POST_ACT),
    # TRAINING, Conv2D-style (D=1, K_d=1) with W tiling and padding.
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-2, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT),
]

# Residual add under a LARGE-MEAN conv output (x_in shifted off zero, so each channel's mean is several
# std). Exercises two precision properties: the order of the batchnorm affine in all modes (deferring
# the mean shift rounds an un-shifted intermediate, losing accuracy in proportion to mean/std — see
# _rescale_phase), and in TRAINING the bn_stats / bn_aggr count-weighted pool, which stays well
# conditioned at any mean where an earlier var = E[Y^2] - E[Y]^2 identity did not.
CONV3D_RESIDUAL_LARGE_MEAN_PARAM_NAMES = CONV3D_BATCH_NORM_PARAM_NAMES + ", x_in_mean"

CONV3D_RESIDUAL_LARGE_MEAN_PARAMS = [
    # EVAL applies the whole affine in one op on the eviction, so it stays accurate at large mean.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, 4.0, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT, 4.0, marks=pytest.mark.fast),
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, 4.0, marks=pytest.mark.fast),
    # BatchNormMode.NONE: no statistics at all, so the eviction residual is exercised cleanly.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.NONE, False, ResidualAddLoc.PRE_ACT, 4.0, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.NONE, False, ResidualAddLoc.POST_ACT, 4.0, marks=pytest.mark.fast),

    # --- TRAINING at large mean: `means` / `variances` are validated fp32 outputs, so these fail loudly if
    # the aggregation regresses. Covered: both residual locations, with / without bias and activation,
    # multiple C_out tiles, both LNC strategies, column tiling. ---

    # float32 (other large-mean rows are bfloat16) is load-bearing: TRAINING stages the raw conv output in
    # an x_in.dtype HBM tensor that bn_stats reads back, so bfloat16 would take statistics over
    # 8-mantissa-bit values — noise entering the variance as (mean/std)^2, percent-level vs 0.000%.

    # x_in_mean is 10.0, not higher, because of the OTHER output `out`: normalizing measures the conv's own
    # fp32 error against std, amplifying it by mean/std. At 20 that pushed `out` past the 1e-5 tolerance on
    # the largest-C_in shapes (statistics still passed at 3e-5%). A row over the `out` tolerance while
    # statistics pass at ~1e-5% is that amplification, so it runs bfloat16.

    # The error in `out` (and only `out`) grows with the conv output's mean/std, so these rows validate
    # `out` at a looser bound scaled by x_in_mean while the statistics stay strict.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, 10.0, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, 10.0, marks=pytest.mark.fast),
    # No activation: the plain rescale path, and PRE_ACT / POST_ACT are equivalent.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, 10.0, marks=pytest.mark.fast),
    # With bias and padding: padded positions join the statistics as f(0 + bias). bfloat16 because in
    # float32 this shape measures 0.0020% on `out` against the 0.001% tolerance (statistics ~1e-5%).
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-5, 0.2, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, 10.0, marks=pytest.mark.fast),
    # C_out > P_MAX: several C_out tiles aggregated independently. This and the C_out=256 row below are
    # squeezed both ways on trn2 — float32 misses the 1e-5 `out` tolerance regardless (0.0016%, a shape
    # property: it fails identically at x_in_mean=0.0), bfloat16 hits the staging limit on `variances`
    # (1.8% vs 1% at mean 10). bfloat16 at mean 4.0 satisfies both. SiLU: GELU is erf in ref, tanh on HW.
    pytest.param(1, 32, 384, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, 4.0, marks=pytest.mark.fast),
    # lnc_shard, shard_on_dh path: each core covers only its D-H subset, and the two halves are
    # pooled by count — the case a "combine two finished means" reduction would get wrong.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, True, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, 10.0, marks=pytest.mark.fast),
    # lnc_shard, C_out-shard path (C_out=256 -> 2 tiles), B=2 with bias. bfloat16 at the lower mean
    # for the same two-sided reason as the C_out=384 case above.
    pytest.param(2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, 1e-3, 0.5, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, 4.0),
    # Conv2D-style (D=1, K_d=1) with W tiling and padding: column tiling packs several D-H bands per
    # flush, so groups span partition bands. bfloat16 — float32 measures 0.0035% on `out` (tol 0.001%).
    pytest.param(1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, 10.0),
    # output_pre_norm: conv_out must stay the raw large-mean conv output.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.PRE_ACT, 10.0, marks=pytest.mark.fast),
]

# BatchNormMode.NONE with a fused residual add, applied on the PSUM eviction beside the plain bias /
# activation path. PRE_ACT with both bias and activation un-fuses the activation from the bias
# tensor_scalar, so it is covered with and without padding.
CONV3D_RESIDUAL_NO_BATCH_NORM_PARAMS = [
    # No bias, no activation: PRE_ACT and POST_ACT are equivalent (plain copy + one add).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    (1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.float32, ResidualAddLoc.POST_ACT),
    # Activation only, both locations (ReLU distinguishes them).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, nl.float32, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # Bias only (tensor_scalar bias + one add).
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, None, False, nl.bfloat16, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # Bias + activation, PRE_ACT: splits the fused activation(bias=...) into bias, add, activation.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.SiLU, False, nl.bfloat16, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # Bias + activation + padding, PRE_ACT: the padded-position fill must run before the add.
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    # Bias + activation + padding, POST_ACT (fused activation(bias=...) kept, then the add).
    (2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, ResidualAddLoc.POST_ACT),
    # C_out > P_MAX (multiple C_out tiles), with bias and stride.
    (1, 64, 256, 4, 16, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, ResidualAddLoc.PRE_ACT),
    # lnc_shard=True (each core adds on its own slice), both sharding shapes.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, True, nl.float32, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    (2, 64, 256, 5, 5, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.POST_ACT),
    # Conv2D-style (D=1, K_d=1) with W tiling and padding.
    (1, 32, 64, 1, 16, 16, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, False, nl.bfloat16, ResidualAddLoc.POST_ACT),
    # Large C_in/C_out where the residual staging tile competes for SBUF: covers the eligible in-place
    # forms (no activation, activation + PRE_ACT) and the non-eligible one (activation + POST_ACT).
    (2, 1024, 1024, 1, 30, 52, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, None, True, nl.bfloat16, ResidualAddLoc.POST_ACT),
    (2, 1024, 1024, 1, 30, 52, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.PRE_ACT),
    (2, 1024, 1024, 1, 30, 52, 3, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.POST_ACT),
    # Fat-filter shapes where the residual staging tile is the largest single SBUF consumer, so these
    # are where recovering the pipeline depth is worth the most.
    (2, 1024, 1024, 1, 60, 104, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.PRE_ACT),
    (2, 1024, 1024, 1, 16, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.PRE_ACT),
    (2, 1024, 1024, 1, 16, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, True, nl.bfloat16, ResidualAddLoc.POST_ACT),
    # Store-batch merging (C_OUT_REP > 1, B > 1, D_out == 1) with a fused residual: the load must mirror
    # the merged store's band layout. Only these are NONE, i.e. merged store with an in-place residual.
    (4, 32, 32, 1, 32, 32, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, False, nl.bfloat16, ResidualAddLoc.PRE_ACT),
    (4, 32, 32, 1, 32, 32, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.ReLU, False, nl.bfloat16, ResidualAddLoc.POST_ACT),
]

CONV3D_RESIDUAL_PARAM_NAMES = CONV3D_PARAM_NAMES + ", residual_add_loc"

# BatchNormMode.NONE with output_pre_norm=True: the flag must be a pure don't-care without a fused
# batchnorm — output equal to the plain conv and no extra outputs (conv3d_torch_ref ignores it too).
# Also confirms the eval-mode mutual-exclusion assertion does not fire in NONE.
CONV3D_OUTPUT_PRE_NORM_NO_FUSE_PARAMS = [
    # No bias, no activation.
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, marks=pytest.mark.fast),
    # With bias, activation and padding (the non-batchnorm eviction path is unaffected).
    pytest.param(2, 32, 64, 4, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.bfloat16, marks=pytest.mark.fast),
]

# Shape for the input-validation (must-raise) tests: they only need to reach _validate_conv3d_inputs,
# so one small config suffices for all of them.
CONV3D_VALIDATION_ERROR_PARAMS = [
    pytest.param(1, 16, 32, 4, 8, 8, 3, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, False, None, False, nl.bfloat16, marks=pytest.mark.fast),
]

# The TRAINING supported-shape limit (a nisa.bn_stats buffer that crowds out the convolution's tiles)
# is covered in test/unit/nkilib/experimental/conv/test_conv3d_supported_shapes.py, which asserts on
# the config builders directly. Those shapes are B=512 at 64x64 / 128x128, so generating their inputs
# and goldens here would allocate 6+ GB on the host to reach a rejection that never runs a kernel.

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

CONV_FUSER_PARAMS = [
    # Default ConvFuser Config with representative batch sizes
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    pytest.param(2, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    # batch=512 is too slow for the fast suite (times out); tag slow so it runs only in the full suite.
    pytest.param(512, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.NONE, marks=pytest.mark.slow),

    # Same ConvFuser configs in BatchNormMode.EVAL (BatchNorm2d.eval(): normalize with the
    # running statistics, no momentum update), across the same batch sizes.
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    pytest.param(2, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    pytest.param(512, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.NONE, marks=pytest.mark.fast),

    # Same ConvFuser configs in training mode with output_pre_norm=True: additionally return the
    # raw pre-batchnorm (and pre-ReLU) convolution output.
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE),
    pytest.param(2, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),
    pytest.param(512, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.NONE, marks=pytest.mark.fast),

    # Same ConvFuser configs with a fused residual add, both insertion points (ReLU distinguishes them)
    # in both batchnorm modes. B=64 exercises the store-batch-merge path.
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT),
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.EVAL, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT),
    pytest.param(1, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.PRE_ACT, marks=pytest.mark.fast),
    pytest.param(64, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, False, ResidualAddLoc.POST_ACT, marks=pytest.mark.fast),
    # Training mode + output_pre_norm with a residual: conv_out must stay residual-free.
    pytest.param(2, 128, 64, 1, 64, 64, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, False, ActFnType.ReLU, True, nl.bfloat16, 1e-5, 0.1, BatchNormMode.TRAINING, True, ResidualAddLoc.PRE_ACT),
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

    @pytest.mark.parametrize(CONV3D_BATCH_NORM_PARAM_NAMES, CONV3D_BATCH_NORM_PARAMS)
    def test_conv3d_fuse_batch_norm(
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
        batch_norm_eps: float,
        momentum: float,
        batch_norm_mode: BatchNormMode,
        output_pre_norm: bool,
        residual_add_loc: ResidualAddLoc,
    ) -> None:
        """Run Conv3D tests with fused batchnorm (mean/variance + normalized output)."""
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
            batch_norm_mode=batch_norm_mode,
            output_pre_norm=output_pre_norm,
            residual_add_loc=residual_add_loc,
            batch_norm_eps=batch_norm_eps,
            momentum=momentum,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_OUTPUT_PRE_NORM_NO_FUSE_PARAMS)
    def test_conv3d_output_pre_norm_without_fuse(
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
        output_pre_norm=True with BatchNormMode.NONE must be a pure don't-care: the output must
        match the plain convolution (+ bias + activation), no extra outputs are returned, and the
        eval-mode mutual-exclusion assertion must not fire.
        """
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
            batch_norm_mode=BatchNormMode.NONE,
            output_pre_norm=True,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_VALIDATION_ERROR_PARAMS)
    def test_conv3d_batch_norm_eval_mode_with_output_pre_norm_raises(
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
        BatchNormMode.EVAL with output_pre_norm=True must be rejected by input validation: eval mode
        normalizes in place on the PSUM eviction, so no pre-batchnorm output exists to return.

        The kernel_assert fires during tracing, which the test framework re-raises wrapped in a
        CompilationException, so this matches on the message rather than the AssertionError type.
        """
        with pytest.raises(Exception, match="mutually exclusive"):
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
                batch_norm_mode=BatchNormMode.EVAL,
                output_pre_norm=True,
            )

    @pytest.mark.parametrize(CONV3D_RESIDUAL_LARGE_MEAN_PARAM_NAMES, CONV3D_RESIDUAL_LARGE_MEAN_PARAMS)
    def test_conv3d_residual_add_large_mean_output(
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
        batch_norm_eps: float,
        momentum: float,
        batch_norm_mode: BatchNormMode,
        output_pre_norm: bool,
        residual_add_loc: ResidualAddLoc,
        x_in_mean: float,
    ) -> None:
        """
        Fused residual add where the convolution output's per-channel mean is many std.

        This is the regime that exercises the batchnorm rescale's precision: normalizing in an order
        that lets an un-shifted intermediate round to the reload buffer's dtype loses accuracy in
        proportion to mean/std, which the zero-mean cases elsewhere in this file cannot detect.
        """
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
            batch_norm_mode=batch_norm_mode,
            output_pre_norm=output_pre_norm,
            batch_norm_eps=batch_norm_eps,
            momentum=momentum,
            residual_add_loc=residual_add_loc,
            x_in_mean=x_in_mean,
        )

    @pytest.mark.parametrize(CONV3D_RESIDUAL_PARAM_NAMES, CONV3D_RESIDUAL_NO_BATCH_NORM_PARAMS)
    def test_conv3d_residual_add_without_batch_norm(
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
        residual_add_loc: ResidualAddLoc,
    ) -> None:
        """Run Conv3D tests with a fused residual add and no batchnorm (BatchNormMode.NONE)."""
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
            residual_add_loc=residual_add_loc,
        )

    @pytest.mark.parametrize(CONV3D_PARAM_NAMES, CONV3D_VALIDATION_ERROR_PARAMS)
    def test_conv3d_residual_add_without_residuals_in_raises(
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
        A residual_add_loc other than NONE without residuals_in must be rejected by input validation.

        The kernel_assert fires during tracing, which the test framework re-raises wrapped in a
        CompilationException, so this matches on the message rather than the AssertionError type.
        """
        with pytest.raises(Exception, match="requires residuals_in"):
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
                residual_add_loc=ResidualAddLoc.PRE_ACT,
                omit_residuals_in=True,
            )

    @pytest.mark.parametrize(CONV3D_BATCH_NORM_PARAM_NAMES, CONV_FUSER_PARAMS)
    def test_conv3d_conv_fuser(
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
        batch_norm_eps: float,
        momentum: float,
        batch_norm_mode: BatchNormMode,
        output_pre_norm: bool,
        residual_add_loc: ResidualAddLoc,
    ) -> None:
        """Run Conv3D tests for ConvFuser configs (fused batchnorm + activation)."""
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
            batch_norm_mode=batch_norm_mode,
            output_pre_norm=output_pre_norm,
            residual_add_loc=residual_add_loc,
            batch_norm_eps=batch_norm_eps,
            momentum=momentum,
            torch_ref=conv_fuser_torch_ref,
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
        batch_norm_mode: BatchNormMode = BatchNormMode.NONE,
        output_pre_norm: bool = False,
        batch_norm_eps: float = 1e-5,
        momentum: float = 0.1,
        residual_add_loc: ResidualAddLoc = ResidualAddLoc.NONE,
        omit_residuals_in: bool = False,
        x_in_mean: float = 0.0,
        torch_ref=conv3d_torch_ref,
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
            residual_add_loc: Where the fused residual add goes relative to the activation
            omit_residuals_in: Skip generating residuals_in to exercise input validation
            x_in_mean: Mean of the generated x_in; non-zero exercises rescale precision
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
                batch_norm_mode=batch_norm_mode,
                output_pre_norm=output_pre_norm,
                batch_norm_eps=batch_norm_eps,
                momentum=momentum,
                residual_add_loc=residual_add_loc,
                omit_residuals_in=omit_residuals_in,
                x_in_mean=x_in_mean,
            )

        def output_tensors(kernel_input):
            # Keys match positionally against the returned tuple (y_out, [conv_out], means, variances,
            # updated_running_means, updated_running_variances), so insertion order must match. EVAL returns only
            # the output; output_pre_norm inserts conv_out right after "out".
            outputs = {
                "out": np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype),
            }
            if batch_norm_mode == BatchNormMode.TRAINING and output_pre_norm:
                # Same shape / dtype as "out"; holds the convolution result before batchnorm.
                outputs["conv_out"] = np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype)
            if batch_norm_mode == BatchNormMode.TRAINING:
                # Batchnorm statistics are per-C_out-channel, always float32.
                outputs["means"] = np.zeros((out_channels,), dtype=np.float32)
                outputs["variances"] = np.zeros((out_channels,), dtype=np.float32)
                # Momentum-updated running statistics are [C_out, 1] float32.
                outputs["updated_running_means"] = np.zeros((out_channels, 1), dtype=np.float32)
                outputs["updated_running_variances"] = np.zeros((out_channels, 1), dtype=np.float32)
            return outputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv3d,
            torch_ref=torch_ref_wrapper(torch_ref),
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
