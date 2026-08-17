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

"""Conv3D supported-shape limits, asserted on the pure config builders without allocating tensors."""

from typing import cast

import nki
import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.allocator import sizeinbytes
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.experimental.conv.conv3d import (
    BatchNormMode,
    ResidualAddLoc,
    _build_conv3d_config,
    _build_memory_config,
    _build_tile_config,
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
        False,
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
