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

"""Conv3D supported-shape limits, the routing to the row-banded path and the plans of the grouped path (with the
depthwise_conv1d route to it), asserted on the pure config builders without allocating tensors."""

import itertools
from typing import cast

import nki
import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.allocator import sizeinbytes
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil
from nkilib_src.nkilib.experimental.conv.conv3d import (
    BatchNormMode,
    Conv3dConfig,
    ResidualAddLoc,
)
from nkilib_src.nkilib.experimental.conv.conv3d_helpers import (
    _build_conv3d_config,
    _build_memory_config,
    _build_tile_config,
)
from nkilib_src.nkilib.experimental.conv.conv3d_kernels import (
    _GROUPED_1D_MAX_TAPS,
    _GROUPED_1D_MIN_CHANNELS,
    _GROUPED_1D_MIN_OUTPUTS,
    _banded_wins,
    _build_conv2d_band_tile_config,
    _grouped_1d_wins,
    _grouped_one_row_sbuf_bytes,
    _grouped_planes_cheaper,
    _plan_grouped_band,
)
from nkilib_src.nkilib.experimental.conv.conv3d_transpose import (
    _banded_transpose_wins,
    _build_conv2d_transpose_banded_config,
    _build_transpose_band_tile_config,
    _transpose_phase_taps,
)


class _ShapeOnlyTensor:
    """
    Stand-in for an nl.NkiTensor: the config builders read only .shape and .dtype.

    Declaring real nl.ndarray handles instead would cost host memory at the shapes under test here
    (measured 4.2 GB for one B=512 128x128 pair), which is the whole reason these limits are checked
    on the builders rather than through the integration harness.
    """

    def __init__(self, shape: tuple[int, ...], dtype):
        self.shape = shape
        self.dtype = dtype


def _as_tensor(stand_in: _ShapeOnlyTensor) -> nl.NkiTensor:
    """Present a shape-only stand-in as the NkiTensor the builders' annotations ask for."""
    return cast(nl.NkiTensor, stand_in)


def _build_configs(batch, in_channels, out_channels, depth, height, width, batch_norm_mode):
    """
    Run the host-side config builders for one shape, returning (cfg, tile_cfg, mem_cfg).

    Wrapped in nki.simulate by the callers below: nl.tile_size needs an active backend to resolve the
    target's SBUF size, and nki.simulate is the public way to get one.
    """
    dtype = nl.bfloat16
    x_in = _ShapeOnlyTensor((batch, in_channels, depth, height, width), dtype)
    filters = _ShapeOnlyTensor((1, 3, 3, in_channels, out_channels), dtype)
    cfg = _build_conv3d_config(
        _as_tensor(x_in),
        _as_tensor(filters),
        None,
        (1, 1, 1),
        (0, 0, 1, 1, 1, 1),
        (1, 1, 1),
        ActFnType.ReLU,
        batch_norm_mode,
        False,
        ResidualAddLoc.NONE,
    )
    dtype_size = sizeinbytes(dtype)
    tile_cfg = _build_tile_config(cfg, dtype_size)
    mem_cfg = _build_memory_config(cfg, tile_cfg, dtype_size)
    return cfg, tile_cfg, mem_cfg


def _build_in_backend(batch, in_channels, out_channels, depth, height, width, batch_norm_mode):
    """Build the configs under a simulator backend, returning a dummy output nki.simulate can accept."""

    def traced():
        _build_configs(batch, in_channels, out_channels, depth, height, width, batch_norm_mode)
        return nl.ndarray(shape=(1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    return nki.simulate(traced)()


# (batch, C_in, C_out, D, H, W) TRAINING shapes whose nisa.bn_stats buffer leaves too little SBUF for
# the convolution's own tiles. The buffer holds ceil(C_out/128) * B * (D-H groups) * (W tiles)
# six-element fp32 groups live for the whole kernel, so it grows with batch AND output spatial size.
# Both of these reserve several times the whole SBUF, so they are refused on every target.
_OVERSIZED_TRAINING_SHAPES = [
    (512, 128, 1024, 1, 64, 64),
    (512, 128, 256, 1, 128, 128),
]


@pytest.mark.parametrize("shape", _OVERSIZED_TRAINING_SHAPES)
@pytest.mark.fast
def test_oversized_training_stats_buffer_is_rejected(shape):
    """
    A TRAINING shape whose statistics buffer crowds out the convolution's tiles must be rejected, with
    that buffer named as the cause rather than only an interleave-cost dump.
    """
    with pytest.raises(AssertionError, match="BatchNormMode.TRAINING is not supported for this shape"):
        _build_in_backend(*shape, BatchNormMode.TRAINING)


@pytest.mark.parametrize("shape", _OVERSIZED_TRAINING_SHAPES)
@pytest.mark.fast
def test_oversized_shapes_are_supported_without_statistics(shape):
    """
    The same shapes build in EVAL and NONE, which accumulate no statistics.

    This is what attributes the TRAINING rejection to the statistics buffer specifically rather than
    to the shape being unbuildable for an unrelated reason.
    """
    for batch_norm_mode in (BatchNormMode.NONE, BatchNormMode.EVAL):
        _build_in_backend(*shape, batch_norm_mode)


@pytest.mark.fast
def test_training_batch_bound_is_exact():
    """
    The supported batch bound is a real boundary: B just under it builds, B just over it is rejected.

    Pins the limit itself rather than the wording of the message. The bound depends on the target's
    SBUF and the rest of the shape, so it is discovered here by bisection instead of hard-coded.
    """
    shape_without_batch = (128, 256, 1, 128, 128)

    def builds(batch):
        try:
            _build_in_backend(batch, *shape_without_batch, BatchNormMode.TRAINING)
            return True
        except AssertionError:
            return False

    assert builds(1), "B=1 must be supported at this shape, else the bisection below is meaningless"
    lo, hi = 1, 512
    assert not builds(hi), "B=512 is expected to exceed the statistics budget at this shape"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if builds(mid):
            lo = mid
        else:
            hi = mid
    assert builds(lo) and not builds(lo + 1), f"bound is not a clean boundary at B={lo}"


def _on_host(decide):
    """Run decide() under a simulator backend (nl.tile_size needs one) and return its result."""
    result = {}

    def traced():
        result["value"] = decide()
        return nl.ndarray(shape=(1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    nki.simulate(traced)()
    return result["value"]


# (C_in, C_out, K, stride, H, W, measured speedup of the banded path over conv3d), 3 x 3 stride 1 unless noted,
# bf16, one trn2 core, filter K x K with padding K // 2 (bench_conv.py: grid h<C>_<W>, u_*, g_*, k_*, sh*_<H>).
_BANDED_MEASURED = [
    (16, 16, 3, 1, 368, 120, 0.50),
    (16, 16, 3, 1, 368, 240, 0.89),
    (16, 16, 3, 1, 368, 480, 1.69),
    (16, 16, 3, 1, 368, 640, 2.31),
    (16, 16, 3, 1, 368, 960, 3.34),
    (16, 16, 3, 1, 368, 1920, 6.17),
    (32, 32, 3, 1, 368, 120, 0.39),
    (32, 32, 3, 1, 368, 240, 0.71),
    (32, 32, 3, 1, 368, 480, 1.36),
    (32, 32, 3, 1, 368, 640, 1.93),
    (32, 32, 3, 1, 368, 960, 2.65),
    (32, 32, 3, 1, 368, 1920, 4.10),
    (64, 64, 3, 1, 368, 120, 0.41),
    (64, 64, 3, 1, 368, 240, 0.78),
    (64, 64, 3, 1, 368, 480, 1.50),
    (64, 64, 3, 1, 368, 640, 2.06),
    (64, 64, 3, 1, 368, 960, 2.57),
    (64, 64, 3, 1, 368, 1920, 3.25),
    (128, 128, 3, 1, 368, 120, 0.23),
    (128, 128, 3, 1, 368, 480, 0.69),
    (128, 128, 3, 1, 368, 640, 1.50),
    (128, 128, 3, 1, 368, 960, 1.04),
    (128, 128, 3, 1, 368, 1920, 1.05),
    (16, 16, 3, 1, 736, 960, 3.34),
    (3, 16, 3, 2, 1472, 1920, 9.4),
    (16, 32, 3, 2, 736, 960, 1.62),
    (32, 64, 3, 2, 368, 480, 0.77),
    (64, 128, 3, 2, 184, 240, 0.45),
    (64, 64, 3, 1, 184, 240, 0.79),
    (128, 128, 3, 1, 92, 120, 0.33),
    (16, 3, 3, 1, 1472, 1920, 12.5),
    (3, 64, 7, 2, 1088, 1920, 12.1),
    (16, 16, 5, 1, 540, 960, 8.0),
    (32, 16, 1, 1, 540, 960, 1.89),
    (8, 32, 3, 1, 1152, 1104, 3.5),
    (64, 64, 3, 1, 92, 480, 1.48),
    (16, 16, 3, 1, 8, 512, 1.02),
    (16, 16, 3, 1, 16, 512, 0.96),
    (16, 16, 3, 1, 32, 512, 1.28),
    (16, 16, 3, 1, 64, 512, 1.74),
    (64, 64, 3, 1, 8, 640, 0.96),
    (64, 64, 3, 1, 16, 640, 1.08),
    (64, 64, 3, 1, 32, 640, 1.77),
]
# The cost model may send a layer to the banded path only when that path won by at least this much.
_BANDED_MIN_ROUTED_SPEEDUP = 1.2
# Layers within this much of the break-even must stay on conv3d.
_BANDED_NO_GAIN = 1.05


def _banded_conv3d_config(
    C_in=16,
    C_out=16,
    K=3,
    stride=1,
    H=128,
    W=640,
    B=1,
    activation_fn=None,
    batch_norm_mode=BatchNormMode.NONE,
    residual_add_loc=ResidualAddLoc.NONE,
    D=1,
    K_d=1,
    dilation=1,
    pad_d=0,
    K_w=None,
    pads=None,
    stride_w=None,
) -> Conv3dConfig:
    """
    A Conv3dConfig for a K x K_w filter (K_w defaults to K) with padding (top, bottom, left, right), by default
    K // 2 on every side, as bench_conv.py builds its layers. stride_w defaults to stride.
    """
    K_w = K if K_w is None else K_w
    stride_w = stride if stride_w is None else stride_w
    pad_top, pad_bottom, pad_left, pad_right = (K // 2, K // 2, K_w // 2, K_w // 2) if pads is None else pads
    H_out = (H + pad_top + pad_bottom - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + pad_left + pad_right - dilation * (K_w - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - (K_d - 1) - 1) + 1
    return Conv3dConfig(
        stride_d=1,
        stride_h=stride,
        stride_w=stride_w,
        pad_d_left=pad_d,
        pad_d_right=pad_d,
        pad_h_top=pad_top,
        pad_h_bottom=pad_bottom,
        pad_w_left=pad_left,
        pad_w_right=pad_right,
        dilation_d=1,
        dilation_h=dilation,
        dilation_w=dilation,
        has_bias=True,
        has_activation=activation_fn is not None,
        activation_fn=activation_fn,
        batch_norm_mode=batch_norm_mode,
        output_pre_norm=False,
        residual_add_loc=residual_add_loc,
        B=B,
        C_in=C_in,
        C_out=C_out,
        D=D,
        H=H,
        W=W,
        K_d=K_d,
        K_h=K,
        K_w=K_w,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
    )


@pytest.mark.fast
def test_routed_layers_are_never_slower_on_measured_points():
    """Every measured layer: routed implies the banded path won clearly, and a layer without a gain is not routed."""

    def decide():
        return [
            _banded_wins(
                _banded_conv3d_config(C_in=c_in, C_out=c_out, K=k, stride=stride, H=h, W=w), nl.bfloat16, False
            )
            for c_in, c_out, k, stride, h, w, _ in _BANDED_MEASURED
        ]

    routed = _on_host(decide)
    for (c_in, c_out, k, stride, h, w, speedup), is_routed in zip(_BANDED_MEASURED, routed, strict=True):
        layer = f"{c_in}->{c_out} k{k} s{stride} {h}x{w}"
        if is_routed:
            assert speedup >= _BANDED_MIN_ROUTED_SPEEDUP, f"{layer} routed to the banded path but measured {speedup}x"
        if speedup < _BANDED_NO_GAIN:
            assert not is_routed, f"{layer} routed to the banded path but measured {speedup}x"
    assert sum(routed) >= 15, "the cost model should still route the wide thin layers"


@pytest.mark.fast
def test_thin_wide_layer_is_routed_and_the_rest_are_not():
    def decide():
        return {
            "thin_wide": _banded_wins(_banded_conv3d_config(), nl.bfloat16, False),
            "fp16": _banded_wins(_banded_conv3d_config(), nl.float16, False),
            "fp32": _banded_wins(_banded_conv3d_config(), nl.float32, False),
            "narrow": _banded_wins(_banded_conv3d_config(W=128), nl.bfloat16, False),
            "few_rows": _banded_wins(_banded_conv3d_config(H=8), nl.bfloat16, False),
            "channels_128": _banded_wins(_banded_conv3d_config(C_in=128, C_out=128, W=960), nl.bfloat16, False),
            "caller_sbm": _banded_wins(_banded_conv3d_config(), nl.bfloat16, True),
            "batch_norm": _banded_wins(_banded_conv3d_config(batch_norm_mode=BatchNormMode.EVAL), nl.bfloat16, False),
            "residual": _banded_wins(
                _banded_conv3d_config(residual_add_loc=ResidualAddLoc.POST_ACT), nl.bfloat16, False
            ),
            "squared_relu": _banded_wins(
                _banded_conv3d_config(activation_fn=ActFnType.SquaredReLU), nl.bfloat16, False
            ),
            "relu": _banded_wins(_banded_conv3d_config(activation_fn=ActFnType.ReLU), nl.bfloat16, False),
            "unequal_strides": _banded_wins(
                _banded_conv3d_config(
                    C_in=16, C_out=16, K=5, K_w=3, stride=2, stride_w=1, H=128, W=640, pads=(2, 1, 0, 2)
                ),
                nl.bfloat16,
                False,
            ),
            "depth": _banded_wins(_banded_conv3d_config(D=4, K_d=3), nl.bfloat16, False),
            "depth_padding": _banded_wins(_banded_conv3d_config(pad_d=1), nl.bfloat16, False),
            "dilated": _banded_wins(_banded_conv3d_config(dilation=2), nl.bfloat16, False),
            "stride_3": _banded_wins(_banded_conv3d_config(stride=3, W=1920), nl.bfloat16, False),
            "filter_9": _banded_wins(_banded_conv3d_config(K=9), nl.bfloat16, False),
            "batch_17": _banded_wins(_banded_conv3d_config(B=17), nl.bfloat16, False),
        }

    decision = _on_host(decide)
    assert decision["thin_wide"] and decision["fp16"] and decision["relu"] and decision["unequal_strides"]
    assert not decision["fp32"]
    for name in (
        "narrow",
        "few_rows",
        "channels_128",
        "caller_sbm",
        "batch_norm",
        "residual",
        "squared_relu",
        "depth",
        "depth_padding",
        "dilated",
        "stride_3",
        "filter_9",
        "batch_17",
    ):
        assert not decision[name], name


@pytest.mark.fast
def test_every_routed_layer_has_a_banded_plan():
    """
    The router must never turn a call conv3d accepts into a failure: every layer it routes, over a grid of channel
    counts, filters, strides, paddings and image sizes up to the routing limits, has a band plan.
    """
    channels = (1, 3, 16, 33, 64)
    filters = ((1, 1), (2, 2), (3, 3), (5, 5), (7, 7), (8, 8), (1, 7), (7, 1), (3, 8))
    layers = list(
        itertools.product(channels, channels, filters, (1, 2), (False, True), (64, 4096), (300, 1024, 4096), (1, 16))
    )

    def decide():
        routed = 0
        unplanned = []
        for c_in, c_out, (k_h, k_w), stride, unpadded, h, w, batch in layers:
            pads = (0, 0, 0, 0) if unpadded else None
            conv3d_cfg = _banded_conv3d_config(
                C_in=c_in, C_out=c_out, K=k_h, K_w=k_w, stride=stride, H=h, W=w, B=batch, pads=pads
            )
            if conv3d_cfg.H_out < 1 or conv3d_cfg.W_out < 1 or not _banded_wins(conv3d_cfg, nl.bfloat16, False):
                continue
            routed += 1
            layer = f"{c_in}->{c_out} k{k_h}x{k_w} s{stride} unpadded={unpadded} {h}x{w} B={batch}"
            # One program (LNC=1, tap fold allowed) and two (LNC=2, no tap fold): both plans must exist.
            for n_prgs in (1, 2):
                try:
                    _build_conv2d_band_tile_config(conv3d_cfg, sizeinbytes(nl.bfloat16), n_prgs)
                except Exception as error:  # noqa: BLE001 - any failure of the plan is a routing bug
                    unplanned.append(f"{layer} n_prgs={n_prgs}: {error}")
        return routed, unplanned

    routed, unplanned = _on_host(decide)
    assert routed >= 100, "the grid should route a good share of its layers"
    assert not unplanned, f"{len(unplanned)} routed layers have no banded plan, first: {unplanned[:3]}"


# (C_in, C_out, K, stride, H, W, tap fold on one program): RGB stems fold filter columns onto the empty contraction
# rows on one program; thin layers with more input channels have no rows to spare.
_TAP_FOLD_LAYERS = [
    (3, 16, 3, 2, 736, 960, 3),
    (3, 64, 7, 2, 544, 960, 4),
    (3, 16, 3, 2, 1472, 1920, 3),
    (16, 16, 3, 1, 368, 960, 1),
    (16, 3, 3, 1, 368, 960, 1),
    (64, 64, 3, 1, 368, 960, 1),
]


@pytest.mark.parametrize("C_in, C_out, K, stride, H, W, fold_one_program", _TAP_FOLD_LAYERS)
def test_tap_fold_on_one_program_only(C_in, C_out, K, stride, H, W, fold_one_program):
    """
    The tap fold is chosen on one program (LNC=1) and is always 1 on two (LNC=2), where the row DMAs, which the fold
    multiplies, set the time; the plans carry the fold they were built with.
    """
    cfg = _banded_conv3d_config(C_in=C_in, C_out=C_out, K=K, stride=stride, H=H, W=W)
    folds = _on_host(
        lambda: [_build_conv2d_band_tile_config(cfg, sizeinbytes(nl.bfloat16), n_prgs).taps_folded for n_prgs in (1, 2)]
    )
    assert folds == [fold_one_program, 1]


# (C_in, C_out, K, H, W, measured speedup of the banded path over conv3d_transpose): stride 2, filter K x K, torch
# padding and output_padding of bench_conv.py (K = 2: 0 / 0, K = 3: 1 / 1, K = 4: 1 / 0), input H x W, bf16, one trn2 core.
_TRANSPOSE_MEASURED = [
    (64, 16, 4, 736, 960, 15.25),
    (64, 16, 4, 368, 576, 10.30),
    (32, 16, 2, 540, 960, 8.96),
    (16, 3, 4, 184, 240, 8.44),
    (64, 16, 4, 184, 480, 7.47),
    (32, 16, 2, 184, 480, 4.43),
    (64, 16, 4, 184, 240, 3.69),
    (8, 8, 4, 92, 240, 3.40),
    (32, 16, 4, 92, 240, 3.36),
    (128, 32, 4, 368, 480, 3.24),
    (64, 16, 4, 32, 240, 3.06),
    (64, 32, 4, 92, 240, 3.06),
    (16, 16, 3, 92, 240, 2.66),
    (3, 3, 2, 92, 240, 2.56),
    (32, 32, 3, 92, 240, 2.40),
    (64, 64, 4, 92, 240, 2.39),
    (64, 16, 4, 184, 120, 2.38),
    (16, 16, 2, 92, 240, 1.94),
    (64, 16, 4, 24, 128, 1.89),
    (32, 16, 2, 32, 240, 1.78),
    (64, 16, 4, 16, 240, 1.74),
    (64, 16, 4, 184, 60, 1.56),
    (32, 16, 2, 184, 120, 1.36),
    (128, 32, 4, 92, 240, 1.30),
    (64, 16, 4, 8, 240, 1.07),
    (32, 16, 2, 16, 240, 1.03),
    (128, 64, 4, 92, 240, 1.00),
    (32, 16, 2, 8, 240, 0.98),
    (64, 16, 4, 4, 240, 0.91),
]
# The same on a logical core of two physical cores (LNC=2), where each core gets half of the work: near the work floor
# both paths sit at the launch floor.
_TRANSPOSE_MEASURED_LNC2 = [
    (64, 16, 4, 736, 960, 14.80),
    (64, 16, 4, 368, 576, 9.93),
    (32, 16, 2, 540, 960, 8.72),
    (64, 16, 4, 184, 480, 6.99),
    (32, 16, 2, 184, 480, 4.00),
    (16, 3, 4, 184, 240, 3.53),
    (64, 16, 4, 184, 240, 3.50),
    (128, 32, 4, 368, 480, 3.12),
    (32, 16, 2, 64, 480, 2.98),
    (8, 8, 4, 92, 240, 2.93),
    (64, 32, 4, 92, 240, 2.87),
    (16, 16, 3, 92, 240, 2.39),
    (32, 32, 3, 92, 240, 2.31),
    (16, 16, 3, 48, 240, 1.84),
    (64, 16, 4, 32, 240, 1.77),
    (64, 16, 4, 24, 240, 1.37),
    (32, 16, 2, 16, 240, 1.05),
    (64, 16, 4, 8, 240, 1.02),
    (64, 16, 4, 16, 240, 1.00),
    (16, 16, 3, 32, 240, 1.00),
    (128, 64, 4, 92, 240, 1.00),
]
_TRANSPOSE_PADDING = {2: (0, 0), 3: (1, 1), 4: (1, 0)}


def _transpose_banded_wins(
    C_in=64,
    C_out=16,
    K=4,
    H=64,
    W=240,
    B=1,
    dtype=nl.bfloat16,
    stride=2,
    pad=None,
    out_pad=None,
    activation_fn=None,
    filter_shape="KDHW_CI_CO",
    has_caller_sbm=False,
    D=1,
    K_d=1,
    stride_d=1,
    dilation=1,
    K_w=None,
    padding=None,
    output_padding=None,
):
    """
    _banded_transpose_wins for a K x K_w filter, torch-style padding and output_padding (K defaults of bench_conv.py).
    padding (pad_d_lo, pad_d_hi, pad_h_lo, pad_h_hi, pad_w_lo, pad_w_hi) and output_padding override pad and out_pad.
    """
    K_w = K if K_w is None else K_w
    default_pad, default_out_pad = _TRANSPOSE_PADDING.get(K, (0, 0))
    pad = default_pad if pad is None else pad
    out_pad = default_out_pad if out_pad is None else out_pad
    padding = (0, 0, pad, pad, pad, pad) if padding is None else padding
    output_padding = (0, out_pad, out_pad) if output_padding is None else output_padding
    return _banded_transpose_wins(
        (B, C_in, D, H, W),
        (K_d, K, K_w, C_in, C_out),
        dtype,
        (stride_d, stride, stride),
        padding,
        (1, dilation, dilation),
        output_padding,
        activation_fn,
        filter_shape,
        has_caller_sbm,
    )


@pytest.mark.fast
@pytest.mark.parametrize("measured", [_TRANSPOSE_MEASURED, _TRANSPOSE_MEASURED_LNC2], ids=["lnc1", "lnc2"])
def test_routed_transposed_layers_are_never_slower_on_measured_points(measured):
    """Every measured layer: routed implies the banded path won clearly, and a layer without a gain is not routed."""

    def decide():
        return [_transpose_banded_wins(C_in=c_in, C_out=c_out, K=k, H=h, W=w) for c_in, c_out, k, h, w, _ in measured]

    routed = _on_host(decide)
    for (c_in, c_out, k, h, w, speedup), is_routed in zip(measured, routed, strict=True):
        layer = f"{c_in}->{c_out} k{k} {h}x{w}"
        if is_routed:
            assert speedup >= _BANDED_MIN_ROUTED_SPEEDUP, f"{layer} routed to the banded path but measured {speedup}x"
        if speedup < _BANDED_NO_GAIN:
            assert not is_routed, f"{layer} routed to the banded path but measured {speedup}x"
    assert sum(routed) >= 12, "the cost model should still route the wide thin layers"


@pytest.mark.fast
def test_thin_transposed_layer_is_routed_and_the_rest_are_not():
    def decide():
        return {
            "thin": _transpose_banded_wins(),
            "fp16": _transpose_banded_wins(dtype=nl.float16),
            "fp32": _transpose_banded_wins(dtype=nl.float32),
            "relu": _transpose_banded_wins(activation_fn=ActFnType.ReLU),
            # the integration tests' layers: K_h != K_w with unequal padding and output_padding, padding past K - 1, lo != hi
            "unequal_padding": _transpose_banded_wins(
                K=4, K_w=2, H=40, W=480, padding=(0, 0, 1, 1, 0, 0), output_padding=(0, 1, 0)
            ),
            "crop_padding": _transpose_banded_wins(K=4, H=40, W=480, pad=4),
            "asymmetric_padding": _transpose_banded_wins(K=4, H=40, W=240, padding=(0, 0, 2, 0, 1, 3)),
            "squared_relu": _transpose_banded_wins(activation_fn=ActFnType.SquaredReLU),
            "co_ci_layout": _transpose_banded_wins(filter_shape="KDHW_CO_CI"),
            "caller_sbm": _transpose_banded_wins(has_caller_sbm=True),
            "depth": _transpose_banded_wins(D=4, K_d=3),
            "stride_d": _transpose_banded_wins(stride_d=2),
            "dilated": _transpose_banded_wins(dilation=2),
            "stride_1": _transpose_banded_wins(stride=1),
            "stride_3": _transpose_banded_wins(stride=3),
            "filter_1": _transpose_banded_wins(K=1),
            "filter_5": _transpose_banded_wins(K=5),
            "wide_cout": _transpose_banded_wins(C_in=64, C_out=96),
            "channels_256": _transpose_banded_wins(C_in=256, C_out=16),
            "batch_17": _transpose_banded_wins(B=17),
            "huge": _transpose_banded_wins(H=4096, W=4096),
            "empty_output": _transpose_banded_wins(K=2, H=1, W=1, pad=1),
        }

    decision = _on_host(decide)
    routed_names = ("thin", "fp16", "relu", "unequal_padding", "crop_padding", "asymmetric_padding")
    for name in routed_names:
        assert decision[name], name
    for name, routed in decision.items():
        if name not in routed_names:
            assert not routed, name


@pytest.mark.fast
def test_every_routed_transposed_layer_has_a_banded_plan():
    """
    The router must never turn a call conv3d_transpose accepts into a failure: every layer it routes, over a grid of
    channel counts, filters, paddings, output paddings and image sizes up to the routing limits, has a banded config
    and band plan.
    """
    channels = (1, 3, 16, 33, 64, 128)
    filters = ((2, 2), (3, 3), (4, 4), (2, 4), (4, 2), (3, 2))
    layers = list(
        itertools.product(channels, channels, filters, (0, 1, 2), (0, 1), (8, 2048), (60, 240, 2048), (1, 16))
    )

    def decide():
        routed = 0
        unplanned = []
        for c_in, c_out, (k_h, k_w), pad, out_pad, h, w, batch in layers:
            if not _transpose_banded_wins(
                C_in=c_in, C_out=c_out, K=k_h, K_w=k_w, pad=pad, out_pad=out_pad, H=h, W=w, B=batch
            ):
                continue
            routed += 1
            layer = f"{c_in}->{c_out} k{k_h}x{k_w} pad={pad} out_pad={out_pad} {h}x{w} B={batch}"
            try:
                cfg = _build_conv2d_transpose_banded_config(
                    _as_tensor(_ShapeOnlyTensor((batch, c_in, h, w), nl.bfloat16)),
                    _as_tensor(_ShapeOnlyTensor((k_h, k_w, c_in, c_out), nl.bfloat16)),
                    None,
                    (2, 2),
                    (pad, pad, pad, pad),
                    (1, 1),
                    (out_pad, out_pad),
                    None,
                    True,
                )
                col_taps = _transpose_phase_taps(cfg.K_w, cfg.stride_w, cfg.pad_w_left, cfg.dilation_w)
                if cfg.H_out < 1 or cfg.W_out < 1:
                    unplanned.append(f"{layer}: empty output")
                for phase in range(min(cfg.stride_w, cfg.W_out)):
                    if len(col_taps[phase]) == 0:
                        unplanned.append(f"{layer}: output column phase {phase} gets no filter column")
                _build_transpose_band_tile_config(cfg, sizeinbytes(nl.bfloat16))
            except Exception as error:  # noqa: BLE001 - any failure of the plan is a routing bug
                unplanned.append(f"{layer}: {error}")
        return routed, unplanned

    routed, unplanned = _on_host(decide)
    assert routed >= 100, "the grid should route a good share of its layers"
    assert not unplanned, f"{len(unplanned)} routed layers have no banded plan, first: {unplanned[:3]}"


# ---------------------------------------------------------------------------------------------------------------
# Grouped path (conv3d with groups > 1) and the depthwise_conv1d route to its one-row diagonal path.
# ---------------------------------------------------------------------------------------------------------------

_PMAX = 128
_PSUM_COLS = 512


def _grouped_out_size(n, pad_lo, pad_hi, k, stride, dilation):
    return (n + pad_lo + pad_hi - dilation * (k - 1) - 1) // stride + 1


def _grouped_plan(B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation):
    """_plan_grouped_band for one layer (pads = (top, bottom, left, right)), as the grouped path calls it."""
    H_out = _grouped_out_size(H, pads[0], pads[1], K_h, stride[0], dilation[0])
    return _on_host(
        lambda: _plan_grouped_band(
            B,
            C_in // groups,
            C_out // groups,
            groups,
            H_out,
            W,
            K_h,
            stride[0],
            stride[1],
            pads[2] + pads[3],
            dilation[0],
            dilation[1] * (K_w - 1),
        )
    )


def _grouped_span(rows, stride_h, K_h, dilation_h):
    return (rows - 1) * stride_h + dilation_h * (K_h - 1) + 1


# (B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation): the layers of the integration test and of the
# grouped-conv benchmarks (MobileNetV2/V3, EfficientNet, ConvNeXt, RepLKNet, ResNeXt, YOLO26, SANA, Evo 2).
_GROUPED_LAYERS = [
    (2, 16, 16, 16, 20, 20, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (2, 32, 32, 4, 20, 20, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (2, 8, 16, 8, 18, 18, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (1, 4, 4, 4, 300, 40, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (2, 16, 16, 16, 21, 21, 3, 3, (2, 2), (1, 1, 1, 1), (1, 1)),
    (2, 8, 8, 8, 24, 24, 3, 3, (1, 1), (2, 2, 2, 2), (2, 2)),
    (16, 96, 96, 96, 56, 56, 7, 7, (1, 1), (3, 3, 3, 3), (1, 1)),
    (32, 144, 144, 144, 56, 56, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (16, 256, 256, 32, 56, 56, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (4, 128, 128, 128, 56, 56, 31, 31, (1, 1), (15, 15, 15, 15), (1, 1)),
    (32, 96, 96, 96, 112, 112, 3, 3, (2, 2), (1, 1, 1, 1), (1, 1)),
    (16, 384, 384, 384, 14, 14, 7, 7, (1, 1), (3, 3, 3, 3), (1, 1)),
    (2, 11200, 11200, 11200, 32, 32, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (16, 256, 256, 256, 20, 20, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)),
    (1, 4096, 4096, 4096, 1, 8192, 1, 7, (1, 1), (0, 0, 6, 6), (1, 1)),
    (1, 4096, 4096, 4096, 1, 8192, 1, 128, (1, 1), (0, 0, 127, 127), (1, 1)),
    (2, 130, 130, 130, 1, 600, 1, 4, (1, 2), (0, 0, 3, 0), (1, 1)),
]


@pytest.mark.parametrize("B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation", _GROUPED_LAYERS)
def test_grouped_plan_fits_and_covers(B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation):
    """Every pass fits the 128 contraction rows and 128 stationary columns, and the blocks cover every output row."""
    plan = _grouped_plan(B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation)
    C_in_g, C_out_g = C_in // groups, C_out // groups
    H_out = _grouped_out_size(H, pads[0], pads[1], K_h, stride[0], dilation[0])
    assert plan.in_parts == plan.g_pack * C_in_g * plan.span <= _PMAX
    assert plan.out_parts == plan.g_pack * C_out_g * plan.rows_out <= _PMAX
    assert plan.span == _grouped_span(plan.rows_out, stride[0], K_h, dilation[0])
    assert plan.n_blocks * plan.rows_out >= H_out
    assert (plan.n_blocks - 1) * plan.rows_out < H_out
    assert plan.n_packs == div_ceil(groups, plan.g_pack)
    assert plan.n_img_groups * plan.n_img >= B
    if plan.seg_out <= _PSUM_COLS:
        assert plan.n_img * plan.seg_out <= _PSUM_COLS
    assert plan.seg_in % stride[1] == 0 and plan.seg_in >= W + pads[2] + pads[3]
    assert plan.tile_cols == plan.n_img * plan.seg_in + dilation[1] * (K_w - 1)


@pytest.mark.parametrize("B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation", _GROUPED_LAYERS)
def test_grouped_plan_takes_the_fewest_passes(B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation):
    """No (groups per pass, rows per block) choice that fits needs fewer passes than the plan's."""
    plan = _grouped_plan(B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation)
    C_in_g, C_out_g = C_in // groups, C_out // groups
    H_out = _grouped_out_size(H, pads[0], pads[1], K_h, stride[0], dilation[0])
    best = None
    for g_pack, rows in itertools.product(range(1, min(groups, _PMAX) + 1), range(1, H_out + 1)):
        span = _grouped_span(rows, stride[0], K_h, dilation[0])
        if g_pack * C_in_g * span <= _PMAX and g_pack * C_out_g * rows <= _PMAX:
            passes = div_ceil(groups, g_pack) * div_ceil(H_out, rows)
            best = passes if best is None else min(best, passes)
    assert plan.n_packs * plan.n_blocks == best


# (layer, expected path): the whole-plane path for small planes with many channels, banded where the band amortizes
_GROUPED_PATHS = [
    ((2, 11200, 11200, 11200, 32, 32, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)), True),  # SANA-1.6B Mix-FFN 1K
    ((16, 256, 256, 256, 20, 20, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)), True),  # YOLO26 P5 head
    ((32, 144, 144, 144, 56, 56, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)), False),  # MobileNetV2 56 x 56
    ((16, 96, 96, 96, 56, 56, 7, 7, (1, 1), (3, 3, 3, 3), (1, 1)), False),  # ConvNeXt stage 1
    ((32, 768, 768, 768, 7, 7, 7, 7, (1, 1), (3, 3, 3, 3), (1, 1)), False),  # ConvNeXt stage 4
    ((4, 128, 128, 128, 56, 56, 31, 31, (1, 1), (15, 15, 15, 15), (1, 1)), False),  # RepLKNet: too many taps
    ((32, 96, 96, 96, 112, 112, 3, 3, (2, 2), (1, 1, 1, 1), (1, 1)), False),  # stride 2: banded only
    ((16, 256, 256, 32, 56, 56, 3, 3, (1, 1), (1, 1, 1, 1), (1, 1)), False),  # grouped (8 per group): banded only
    ((1, 4096, 4096, 4096, 1, 8192, 1, 7, (1, 1), (0, 0, 6, 6), (1, 1)), False),  # 1-D: the one-row path
]


@pytest.mark.parametrize("layer, planes", _GROUPED_PATHS)
def test_grouped_planes_choice(layer, planes):
    """The cost model sends small planes with many channels to the whole-plane path and keeps the rest banded."""
    B, C_in, C_out, groups, H, W, K_h, K_w, stride, pads, dilation = layer
    plan = _grouped_plan(*layer)
    H_out = _grouped_out_size(H, pads[0], pads[1], K_h, stride[0], dilation[0])
    W_out = _grouped_out_size(W, pads[2], pads[3], K_w, stride[1], dilation[1])
    got = _on_host(
        lambda: _grouped_planes_cheaper(
            plan, B, C_in, H, W, H_out, W_out, K_h, K_w, C_in // groups, C_out // groups, stride, pads, nl.bfloat16
        )
    )
    assert got == planes


def _grouped_1d_routes(N, C, W, S, stride_w, pad_l, pad_r, dtype=nl.bfloat16, pad_h=(0, 0), filter_dtype=None):
    img = _ShapeOnlyTensor((N, C, 1, W), dtype)
    filt = _ShapeOnlyTensor((C, 1, 1, S), dtype if filter_dtype is None else filter_dtype)
    return _on_host(lambda: _grouped_1d_wins(_as_tensor(img), _as_tensor(filt), (pad_h, (pad_l, pad_r)), (1, stride_w)))


# (N, C, W, S, stride_w, pad_l, pad_r, dtype, measured speedup of the one-row diagonal path over
# depthwise_conv1d_implicit_gemm's own kernel), trn2, LNC=2, back-to-back calls. The suite's small layers run at the
# launch floor (about 0.09 ms) on both kernels.
_GROUPED_1D_MEASURED = [
    (1, 1536, 2048, 4, 1, 3, 0, nl.bfloat16, 40.8),  # Mamba causal conv1d (d_inner 1536, d_conv 4)
    (8, 1536, 2048, 4, 1, 3, 0, nl.bfloat16, 189.8),
    (8, 512, 1000, 31, 1, 15, 15, nl.bfloat16, 54.9),  # Conformer convolution module
    (1, 12288, 8192, 3, 1, 2, 0, nl.bfloat16, 191.9),  # Evo 2 7B featurizer
    (1, 4096, 8192, 7, 1, 6, 0, nl.bfloat16, 119.2),  # Evo 2 Hyena-SE
    (1, 4096, 8192, 128, 1, 127, 0, nl.bfloat16, 38.9),  # Evo 2 Hyena-MR
    (1, 512, 5000, 8, 1, 0, 0, nl.bfloat16, 21.7),  # the suite's FS-ASR layer
    (1, 256, 4096, 64, 2, 0, 0, nl.bfloat16, 5.1),
    (1, 256, 4096, 64, 1, 0, 0, nl.float32, 3.6),
    # long sequences; past the SBUF fit the diagonal path still compiled and won here, but is not routed
    (1, 256, 32768, 4, 1, 3, 0, nl.bfloat16, 61.0),
    (1, 256, 49152, 4, 1, 3, 0, nl.bfloat16, 97.9),
    (1, 256, 65536, 4, 1, 3, 0, nl.bfloat16, 46.5),  # past the SBUF fit
    (1, 256, 16384, 4, 1, 3, 0, nl.float32, 43.6),
    (1, 256, 24576, 4, 1, 3, 0, nl.float32, 46.7),
    (1, 256, 32768, 4, 1, 3, 0, nl.float32, 37.2),  # past the SBUF fit
    (1, 256, 16384, 128, 1, 127, 0, nl.bfloat16, 16.5),
    (1, 256, 24576, 128, 1, 127, 0, nl.bfloat16, 11.2),
    (1, 256, 32768, 128, 1, 127, 0, nl.bfloat16, 12.5),
    (1, 256, 65536, 128, 1, 127, 0, nl.bfloat16, 11.0),  # past the SBUF fit
    (1, 256, 8192, 128, 1, 127, 0, nl.float32, 7.44),
    (1, 256, 12288, 128, 1, 127, 0, nl.float32, 4.42),
    (1, 256, 16384, 128, 1, 127, 0, nl.float32, 5.20),
    (1, 256, 32768, 128, 1, 127, 0, nl.float32, 4.25),  # past the SBUF fit
    (1, 256, 4096, 256, 1, 0, 0, nl.bfloat16, 2.13),
    (1, 4096, 8192, 256, 1, 255, 0, nl.bfloat16, 35.2),
    (1, 256, 1024, 512, 1, 0, 0, nl.bfloat16, 2.07),
    (1, 16, 512, 200, 1, 0, 0, nl.bfloat16, 0.96),
    (1, 16, 512, 3, 1, 0, 0, nl.bfloat16, 0.92),
    (1, 16, 512, 64, 1, 0, 0, nl.bfloat16, 1.02),
    (1, 2, 512, 64, 1, 0, 0, nl.bfloat16, 0.97),
    (1, 256, 512, 64, 1, 0, 0, nl.bfloat16, 2.01),
    (1, 256, 512, 64, 2, 5, 5, nl.bfloat16, 59.1),
    (1, 256, 256, 128, 1, 0, 0, nl.bfloat16, 1.11),
    (1, 1024, 1024, 128, 1, 0, 0, nl.bfloat16, 2.83),
    (4, 256, 512, 128, 1, 0, 0, nl.bfloat16, 3.56),
    (1, 256, 2048, 128, 1, 10, 10, nl.bfloat16, 14.1),
    (1, 256, 512, 64, 1, 0, 0, nl.float32, 2.35),
    (1, 256, 512, 128, 1, 0, 0, nl.float16, 1.63),
]
# Routed layers must have measured at least this much faster; a layer measured slower must not be routed.
_GROUPED_1D_MIN_ROUTED_SPEEDUP = 1.1
# Layers that failed to compile on the one-row diagonal path (a tile past the SBUF partition size).
_GROUPED_1D_FAILS_ON_DIAGONAL = [
    (1, 256, 131072, 4, 1, 3, 0, nl.bfloat16),
    (1, 256, 262144, 4, 1, 3, 0, nl.bfloat16),
    (1, 256, 65536, 4, 1, 3, 0, nl.float32),
    (1, 256, 4096, 1024, 1, 0, 0, nl.bfloat16),
]


def test_grouped_1d_route_on_measured_points():
    """Every measured layer: routed implies the diagonal path won, and a layer it lost on is not routed."""
    routed = 0
    for N, C, W, S, stride_w, pad_l, pad_r, dtype, speedup in _GROUPED_1D_MEASURED:
        layer = f"N={N} C={C} W={W} S={S} s={stride_w} pad=({pad_l}, {pad_r}) {dtype}"
        is_routed = _grouped_1d_routes(N, C, W, S, stride_w, pad_l, pad_r, dtype)
        if is_routed:
            routed += 1
            assert speedup >= _GROUPED_1D_MIN_ROUTED_SPEEDUP, (
                f"{layer} routed to the diagonal path but measured {speedup}x"
            )
        if speedup < 1.0:
            assert not is_routed, f"{layer} routed to the diagonal path but measured {speedup}x"
    assert routed >= 28, "the sequence-model layers should stay routed"


@pytest.mark.parametrize("N, C, W, S, stride_w, pad_l, pad_r, dtype", _GROUPED_1D_FAILS_ON_DIAGONAL)
def test_grouped_1d_route_skips_layers_that_do_not_fit(N, C, W, S, stride_w, pad_l, pad_r, dtype):
    """Layers whose tiles do not fit SBUF stay on the original kernel."""
    assert not _grouped_1d_routes(N, C, W, S, stride_w, pad_l, pad_r, dtype)


@pytest.mark.parametrize(
    "N, C, W, S, stride_w, pad_l, pad_r, kwargs",
    [
        (1, 256, 512, _GROUPED_1D_MAX_TAPS + 1, 1, 0, 0, {}),  # longer than any filter measured
        (1, 8192, 4096, 4093, 1, 0, 0, {}),  # the suite's Qwen3 layer: 4 outputs of a 4093-tap filter
        (1, 256, 129, 128, 1, 0, 0, {}),  # 2 outputs
        (1, 256, 20, 8, 1, 0, 0, {}),  # 13 outputs, under _GROUPED_1D_MIN_OUTPUTS
        (1, _GROUPED_1D_MIN_CHANNELS - 1, 4096, 8, 1, 0, 0, {}),  # too few channels
        (1, 256, 4096, 512, 1, 0, 0, {"dtype": nl.float32}),  # fp32 stationaries of 512 taps: past SBUF
        (1, 256, 512, 8, 1, 0, 0, {"pad_h": (1, 0)}),  # padding of the unit height axis
        (1, 256, 512, 8, 1, 0, 0, {"filter_dtype": nl.float32}),  # mixed dtypes
    ],
)
def test_grouped_1d_route_keeps_the_rest(N, C, W, S, stride_w, pad_l, pad_r, kwargs):
    """Long filters, short outputs, few channels and options the diagonal path does not take stay put."""
    assert not _grouped_1d_routes(N, C, W, S, stride_w, pad_l, pad_r, **kwargs)


def test_every_routed_grouped_1d_call_has_a_one_row_plan():
    """
    Over a grid of batch, channel, width, tap, stride and padding choices, every routed call plans one row per pass
    (the diagonal path) and its tiles fit the SBUF budget, so routing never turns a call the original kernel takes
    into a failure.
    """
    grid = itertools.product(
        (1, 3),
        (128, 300, 4096),
        (64, 1000, 8192, 40000),
        (1, 4, 31, 128, 300, 512),
        (1, 2, 5),
        ((0, 0), (3, 0), (15, 15)),
        (nl.bfloat16, nl.float32),
    )

    def decide():
        routed = 0
        bad = []
        budget = nl.tile_size.total_available_sbuf_size
        for N, C, W, S, stride_w, (pad_l, pad_r), dtype in grid:
            if (W + pad_l + pad_r - S) // stride_w + 1 < 1:
                continue
            img = _as_tensor(_ShapeOnlyTensor((N, C, 1, W), dtype))
            filt = _as_tensor(_ShapeOnlyTensor((C, 1, 1, S), dtype))
            if not _grouped_1d_wins(img, filt, ((0, 0), (pad_l, pad_r)), (1, stride_w)):
                continue
            routed += 1
            plan = _plan_grouped_band(N, 1, 1, C, 1, W, 1, 1, stride_w, pad_l + pad_r, 1, S - 1)
            dsize = 4 if dtype == nl.float32 else 2
            if not (plan.span == 1 and plan.rows_out == 1 and plan.n_packs == div_ceil(C, _PMAX)):
                bad.append(f"N={N} C={C} W={W} S={S} s={stride_w}: not a one-row plan")
            if _grouped_one_row_sbuf_bytes(plan, S, dsize) > budget:
                bad.append(f"N={N} C={C} W={W} S={S} s={stride_w}: tiles past the SBUF budget")
        return routed, bad

    routed, bad = _on_host(decide)
    assert routed >= 200, "the grid should route a good share of its layers"
    assert not bad, f"{len(bad)} routed layers without a one-row plan that fits, first: {bad[:3]}"


def test_grouped_1d_route_constants():
    """The routing limits the docstrings promise."""
    assert _GROUPED_1D_MAX_TAPS == 512
    assert _GROUPED_1D_MIN_CHANNELS == 128
    assert _GROUPED_1D_MIN_OUTPUTS == 16
