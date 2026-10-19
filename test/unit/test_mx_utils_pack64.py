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

"""Unit tests for the PACK64 gate/up MX scale remap and its inverse (test/utils/mx_utils.py)."""

import numpy as np
import pytest

from test.utils.mx_utils import pack64_gate_up_scale, unpack64_gate_up_scale


@pytest.mark.parametrize("E, n_h512, I", [(1, 1, 512), (3, 1, 1024), (2, 6, 3072)])
def test_unpack64_inverts_pack64(E, n_h512, I):
    rng = np.random.default_rng(42)
    scale = rng.integers(0, 256, size=(E, 16, 2, n_h512, I), dtype=np.uint8)
    packed = pack64_gate_up_scale(scale)
    assert packed.shape == (E, 2, 64, n_h512, I // 4)
    unpacked = unpack64_gate_up_scale(packed)
    assert unpacked.shape == scale.shape
    assert unpacked.flags.c_contiguous
    np.testing.assert_array_equal(unpacked, scale)


def test_unpack64_passes_through_stacked_leading_dims():
    rng = np.random.default_rng(0)
    scale = rng.integers(0, 256, size=(2, 3, 16, 2, 1, 512), dtype=np.uint8)  # [L, E, 16, 2, H/512, I]
    np.testing.assert_array_equal(unpack64_gate_up_scale(pack64_gate_up_scale(scale)), scale)


def test_pack64_element_mapping_matches_contract():
    """Data free index i = tile_i*512 + k*128 + j lands at partition 16*q + 4*k + r, free (h, tile_i*128 + j)."""
    I = 1024
    scale = np.arange(16 * 2 * I, dtype=np.int64).reshape(1, 16, 2, 1, I)
    packed = pack64_gate_up_scale(scale)
    for p, g, i in [(0, 0, 0), (5, 1, 700), (15, 1, I - 1), (9, 0, 513)]:
        q, r = divmod(p, 4)
        tile_i, rem = divmod(i, 512)
        k, j = divmod(rem, 128)
        assert packed[0, g, 16 * q + 4 * k + r, 0, tile_i * 128 + j] == scale[0, p, g, 0, i]
