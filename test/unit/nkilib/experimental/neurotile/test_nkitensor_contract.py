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

"""Drift guard for neurotile's coupling to private ``NkiTensor`` internals.

NkiTensor ships in an external wheel. neurotile relies on a few NkiTensor
behaviors that are not part of its public contract; this test pins them so an
upstream change fails *here* (a fast pure-Python unit test) instead of silently
miscompiling on device. The two confined access points are:

  - ``_helpers.nki_strided_view`` -- builds a memory-less NkiTensor with custom
    strides via the private ``NkiTensor._copy(shape=, strides=)``.
  - ``_helpers.physical_row_width`` -- reads the private ``_storage_shape`` to
    get the physical free width (the AP partition stride).

If any assertion here fails after an ``nki`` wheel bump, the private contract
drifted: fix the adapter in ``_helpers`` (one site each), do not scatter the fix.

Pure Python -- no NKI tracer, no device.
"""

import nki.language as nl
import pytest
from nki.language.tensor import NkiTensor
from nkilib_src.nkilib.experimental.neurotile.core._helpers import (
    nki_strided_view,
    physical_row_width,
)

from test.utils.pytest_test_metadata import pytest_marks


def _real(shape, buffer=nl.sbuf):
    """A real (storage-backed) NkiTensor, for the _storage_shape contract."""
    return NkiTensor(shape=tuple(shape), dtype="float32", storage=object(), buffer=buffer)


@pytest_marks(["neurotile"])
class TestNkiStridedViewContract:
    """`nki_strided_view` (and thus `NkiTensor._copy(shape=, strides=)`)."""

    @pytest.mark.fast
    def test_carries_shape_and_strides(self):
        v = nki_strided_view((128, 256), (256, 1), "float32", nl.shared_hbm)
        assert tuple(v.shape) == (128, 256)
        assert tuple(v.strides) == (256, 1)

    @pytest.mark.fast
    def test_accepts_logical_strides_ap_would_reject(self):
        # The whole reason _copy is used: a logical partition stride (512) that
        # differs from the physical free width. .ap() rejects this; _copy must not.
        v = nki_strided_view((128, 512), (512, 1), "float32", nl.sbuf)
        assert tuple(v.strides) == (512, 1)

    @pytest.mark.fast
    def test_transforms_run_on_it(self):
        # Every NDSlice transform routes through this view; the native ops must
        # produce the expected (shape, strides) and stay offset-preserving.
        v = nki_strided_view((128, 256), (256, 1), "float32", nl.shared_hbm)
        for op, want in [
            (v.reshape((256, 128)), ((256, 128), (128, 1))),
            (v.reshape_dim(1, (8, 32)), ((128, 8, 32), (256, 32, 1))),
            (v.permute((1, 0)), ((256, 128), (1, 256))),
            (v.expand_dim(1), ((128, 1, 256), None)),  # stride[1] don't-care (size-1)
        ]:
            assert (tuple(op.shape), None if want[1] is None else tuple(op.strides)) == (
                want[0],
                want[1],
            )
            assert op.offset == 0, (
                "transform must be offset-preserving (NDSlice._rebuild_from reads only shape/strides)"
            )


@pytest_marks(["neurotile"])
class TestPhysicalRowWidthContract:
    """`physical_row_width` (and thus the `_storage_shape` read)."""

    @pytest.mark.fast
    def test_top_level(self):
        assert physical_row_width(_real((128, 512))) == 512

    @pytest.mark.fast
    def test_survives_column_slice(self):
        # The case that matters: a sub-width slice still reports the PARENT's
        # physical row width (this is why we read _storage_shape, not shape).
        sliced = _real((128, 512))[:, 0:128]
        assert physical_row_width(sliced) == 512

    @pytest.mark.fast
    def test_multi_free_dim(self):
        assert physical_row_width(_real((128, 4, 64))) == 256

    @pytest.mark.fast
    def test_equals_get_pattern_partition_stride_when_partition_preserved(self):
        # Documented coincidence we rely on at the call sites (partition dim
        # untouched): physical row width == the view's dim-0 stride. (NOT true
        # for a permuted/reshaped view -- those never reach physical_row_width.)
        for t in (_real((128, 512)), _real((128, 512))[:, 128:256], _real((128, 4, 64))):
            assert physical_row_width(t) == t.get_pattern()[0][0]
