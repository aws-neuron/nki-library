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

from typing import Any, Optional, final

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.experimental.conv.conv3d_transpose import (
    _banded_transpose_wins,
    _conv3d_transpose_banded,
    conv3d_transpose,
)
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
    bias_shape: Optional[tuple[int, ...]] = None,
    pad_h_hi: Optional[int] = None,
    pad_w_hi: Optional[int] = None,
):
    """
    Generate kernel inputs for a Conv3DTranspose call.

    Builds the filter in a forward-conv layout with (C_in=in_channels,
    C_out=out_channels). When flip_filter_taps_in_kernel is True the kernel reads spatially-reversed taps internally.
    If filter_shape == "KDHW_CO_CI", the channel axes are also swapped so
    position 3 holds C_out and position 4 holds C_in. bias_shape defaults to [C_out]; the kernel reads the
    first C_out elements of any bias, e.g. [C_out, 1] or longer. pad_h / pad_w are the low-side paddings;
    pad_h_hi / pad_w_hi, when given, are the high sides (default: the same as the low side).
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

    bias = generate_tensor(name="bias", shape=bias_shape or (out_channels,), dtype=dtype) if use_bias else None

    return {
        "x_in": x_in,
        "filters": kernel_filter,
        "bias": bias,
        "stride": (stride_d, stride_h, stride_w),
        "padding": (
            pad_d,
            pad_d,
            pad_h,
            pad_h if pad_h_hi is None else pad_h_hi,
            pad_w,
            pad_w if pad_w_hi is None else pad_w_hi,
        ),
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
# Thin 2D stride-2 upsamplers with wide rows that conv3d_transpose routes to the row-banded path on trn2 (expect_banded,
# conv3d_transpose._banded_transpose_wins), with the asymmetric cases of the routed wrapper (K_h != K_w, pad_h != pad_w,
# output_padding h != w, padding > K - 1), and twins that stay on the embedded conv3d core.
CONV3D_TRANSPOSE_BANDED_ROUTE_PARAMS = [
    pytest.param(1, 64, 16, 1, 40, 160, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, True, marks=pytest.mark.fast),
    pytest.param(2, 32, 16, 1, 32, 400, 1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1, True, ActFnType.ReLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, True, marks=pytest.mark.fast),
    (1, 16, 16, 1, 48, 240, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, "KDHW_CI_CO", nl.bfloat16, 0, 1, 1, True, True),
    (1, 128, 32, 1, 32, 480, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, False, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, True),
    pytest.param(1, 64, 16, 1, 40, 480, 1, 4, 2, 1, 2, 2, 0, 1, 0, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 1, 0, True, True, marks=pytest.mark.fast),
    (1, 64, 16, 1, 40, 480, 1, 4, 4, 1, 2, 2, 0, 4, 4, 1, 1, 1, True, ActFnType.ReLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, True),
    (1, 64, 96, 1, 8, 24, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, False),
    (1, 64, 16, 1, 4, 24, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, False),
]
# Layers the router keeps on the embedded conv3d core, run on the row-banded path directly so its corner cases stay
# covered. The two trailing fields are pad_h_hi and pad_w_hi.
CONV3D_TRANSPOSE_BANDED_DIRECT_PARAMS = [
    pytest.param(1, 16, 8, 1, 8, 12, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, 1, 1, marks=pytest.mark.fast),
    pytest.param(2, 8, 8, 1, 6, 10, 1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1, False, ActFnType.ReLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, 0, 0, marks=pytest.mark.fast),
    (1, 4, 6, 1, 7, 9, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, ActFnType.SiLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, 1, 1),
    (1, 64, 32, 1, 8, 16, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, ActFnType.GELU, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, 1, 1),
    (1, 64, 64, 1, 6, 12, 1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, 0, 0),
    (1, 16, 16, 1, 8, 12, 1, 3, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, ActFnType.ReLU, False, "KDHW_CI_CO", nl.float16, 0, 1, 1, False, 1, 1),
    (1, 3, 8, 1, 1, 1, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, 1, 1),
    # Asymmetric padding: (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) differ.
    pytest.param(1, 8, 8, 1, 7, 8, 1, 4, 4, 1, 2, 2, 0, 2, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, True, 0, 3, marks=pytest.mark.fast),
    (1, 6, 10, 1, 9, 5, 1, 3, 3, 1, 2, 2, 0, 0, 2, 1, 1, 1, True, ActFnType.SiLU, False, "KDHW_CI_CO", nl.bfloat16, 0, 1, 0, False, 1, 0),
    # Output width above the SBUF window of one pass: the column-group windows are walked in turn.
    (1, 64, 64, 1, 2, 12000, 1, 4, 4, 1, 2, 2, 0, 1, 1, 1, 1, 1, True, None, False, "KDHW_CI_CO", nl.bfloat16, 0, 0, 0, False, 1, 1),
]
# A thin stride-2 layer with wide rows, routed to the row-banded path on trn2; the arguments of _run_conv3d_transpose.
_ROUTED_LAYER: dict[str, Any] = {
    "batch": 1, "in_channels": 64, "out_channels": 16, "depth": 1, "height": 40, "width": 160,
    "filter_d": 1, "filter_h": 4, "filter_w": 4, "stride_d": 1, "stride_h": 2, "stride_w": 2,
    "pad_d": 0, "pad_h": 1, "pad_w": 1, "dilation_d": 1, "dilation_h": 1, "dilation_w": 1,
    "use_bias": True, "activation_fn": None, "lnc_shard": False, "filter_shape": "KDHW_CI_CO", "dtype": nl.bfloat16,
    "output_pad_d": 0, "output_pad_h": 0, "output_pad_w": 0, "flip_filter_taps_in_kernel": True,
}
# fmt: on


def _conv3d_transpose_banded_kernel(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    output_padding: tuple[int, int, int] = (0, 0, 0),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
    filter_shape: str = "KDHW_CI_CO",
    flip_filter_taps_in_kernel: bool = False,
    sbm=None,
    use_auto_allocation: bool = False,
):
    """conv3d_transpose's row-banded path called directly, with conv3d_transpose's arguments (KDHW_CI_CO filters)."""
    assert filter_shape == "KDHW_CI_CO" and sbm is None  # noqa: S101
    return _conv3d_transpose_banded(
        x_in, filters, bias, stride, padding, dilation, output_padding, activation_fn, flip_filter_taps_in_kernel
    )


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
        bias_shape: Optional[tuple[int, ...]] = None,
        pad_h_hi: Optional[int] = None,
        pad_w_hi: Optional[int] = None,
        kernel_entry=conv3d_transpose,
    ) -> None:
        """Run a Conv3DTranspose kernel test against a torch ConvTranspose3d reference."""
        pad_h_total = pad_h + (pad_h if pad_h_hi is None else pad_h_hi)
        pad_w_total = pad_w + (pad_w if pad_w_hi is None else pad_w_hi)
        D_out = (depth - 1) * stride_d + dilation_d * (filter_d - 1) - 2 * pad_d + output_pad_d + 1
        H_out = (height - 1) * stride_h + dilation_h * (filter_h - 1) - pad_h_total + output_pad_h + 1
        W_out = (width - 1) * stride_w + dilation_w * (filter_w - 1) - pad_w_total + output_pad_w + 1

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
                bias_shape=bias_shape,
                pad_h_hi=pad_h_hi,
                pad_w_hi=pad_w_hi,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, out_channels, D_out, H_out, W_out), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_entry,
            torch_ref=torch_ref_wrapper(conv3d_transpose_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        # bfloat16 / float16 results are rounded once from fp32 PSUM sums.
        tol = 1e-5
        if dtype == nl.bfloat16:
            tol = 1e-2
        elif dtype == nl.float16:
            tol = 2e-3
        rtol = tol
        atol = tol

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

    @pytest.mark.parametrize(CONV3D_TRANSPOSE_PARAM_NAMES + ", expect_banded", CONV3D_TRANSPOSE_BANDED_ROUTE_PARAMS)
    def test_conv3d_transpose_banded_route(
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
        expect_banded: bool,
    ) -> None:
        """Thin stride-2 upsamplers (routed to the row-banded path on trn2) and their twins."""
        result = {}

        def decide():
            result["routed"] = _banded_transpose_wins(
                (batch, in_channels, depth, height, width),
                (filter_d, filter_h, filter_w, in_channels, out_channels),
                dtype,
                (stride_d, stride_h, stride_w),
                (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w),
                (dilation_d, dilation_h, dilation_w),
                (output_pad_d, output_pad_h, output_pad_w),
                activation_fn,
                filter_shape,
                False,
            )
            return nl.ndarray(shape=(1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

        nki.simulate(decide)()  # nl.tile_size needs a backend
        routed = result["routed"]
        assert routed == expect_banded, f"_banded_transpose_wins returned {routed}, expected {expect_banded}"  # noqa: S101
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

    @pytest.mark.parametrize(
        CONV3D_TRANSPOSE_PARAM_NAMES + ", pad_h_hi, pad_w_hi", CONV3D_TRANSPOSE_BANDED_DIRECT_PARAMS
    )
    def test_conv3d_transpose_banded_direct(
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
        pad_h_hi: int,
        pad_w_hi: int,
    ) -> None:
        """The row-banded path on layers the router does not send it (trn2 only, like the routing)."""
        if platform_target != Platforms.TRN2:
            pytest.skip("conv3d_transpose runs the row-banded path on trn2 only")
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
            pad_h_hi=pad_h_hi,
            pad_w_hi=pad_w_hi,
            kernel_entry=_conv3d_transpose_banded_kernel,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("bias_shape", [(16, 1), (24,)], ids=["column", "longer"])
    def test_conv3d_transpose_banded_route_bias_shape(
        self, test_manager: Orchestrator, platform_target: Platforms, bias_shape: tuple[int, ...]
    ) -> None:
        """conv3d_transpose reads the first C_out elements of the bias without checking its shape; so must the routed path."""
        self._run_conv3d_transpose(
            test_manager=test_manager, platform_target=platform_target, **_ROUTED_LAYER, bias_shape=bias_shape
        )

    @pytest.mark.fast
    def test_conv3d_transpose_banded_route_asymmetric_padding(
        self, test_manager: Orchestrator, platform_target: Platforms
    ) -> None:
        """padding (lo != hi) reaches the routed path in the right order; the reference crops the unpadded output."""
        layer: dict[str, Any] = {**_ROUTED_LAYER, "width": 240, "pad_h": 2, "pad_h_hi": 0, "pad_w": 1, "pad_w_hi": 3}
        self._run_conv3d_transpose(test_manager=test_manager, platform_target=platform_target, **layer)
