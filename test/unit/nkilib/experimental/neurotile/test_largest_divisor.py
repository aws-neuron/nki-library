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

"""CPU-only unit tests for nt.largest_divisor() and nt.ceiling_div() trace-time utilities."""

import pytest
from nkilib_src.nkilib.experimental import neurotile as nt
from nkilib_src.nkilib.experimental.neurotile.core._helpers import p_tile_count

from test.utils.pytest_test_metadata import pytest_marks


@pytest_marks(["neurotile"])
class TestLargestDivisor:
    @pytest.mark.fast
    def test_exact_divisor(self):
        assert nt.largest_divisor(24, 8) == 8

    def test_no_exact_match(self):
        assert nt.largest_divisor(7, 4) == 1

    def test_max_equals_n(self):
        assert nt.largest_divisor(6, 6) == 6

    def test_max_larger_than_n(self):
        assert nt.largest_divisor(4, 16) == 4

    def test_prime_number(self):
        assert nt.largest_divisor(13, 8) == 1

    def test_one(self):
        assert nt.largest_divisor(1, 8) == 1

    def test_n_equals_max(self):
        assert nt.largest_divisor(8, 8) == 8

    def test_large_values(self):
        assert nt.largest_divisor(1024, 8) == 8
        assert nt.largest_divisor(1024, 16) == 16
        assert nt.largest_divisor(1024, 512) == 512

    def test_two(self):
        assert nt.largest_divisor(2, 8) == 2

    def test_common_kernel_sizes(self):
        # K_tiles=8, max=8 -> 8
        assert nt.largest_divisor(8, 8) == 8
        # K_tiles=6, max=8 -> 6
        assert nt.largest_divisor(6, 8) == 6
        # K_tiles=7, max=8 -> 7
        assert nt.largest_divisor(7, 8) == 7
        # I_tiles=16, max=8 -> 8
        assert nt.largest_divisor(16, 8) == 8
        # I_tiles=3, max=8 -> 3
        assert nt.largest_divisor(3, 8) == 3


@pytest_marks(["neurotile"])
class TestLargestDivisorMisuseGuards:
    def test_n_must_be_positive(self):
        with pytest.raises(AssertionError, match="n must be a positive int"):
            nt.largest_divisor(0, 8)
        with pytest.raises(AssertionError, match="n must be a positive int"):
            nt.largest_divisor(-1, 8)

    def test_n_must_be_int(self):
        with pytest.raises(AssertionError, match="n must be a positive int"):
            nt.largest_divisor(8.0, 8)

    def test_max_val_must_be_positive(self):
        with pytest.raises(AssertionError, match="max_val must be a positive int"):
            nt.largest_divisor(8, 0)
        with pytest.raises(AssertionError, match="max_val must be a positive int"):
            nt.largest_divisor(8, -1)

    def test_max_val_must_be_int(self):
        with pytest.raises(AssertionError, match="max_val must be a positive int"):
            nt.largest_divisor(8, 4.0)


@pytest_marks(["neurotile"])
class TestCeilingDivBasic:
    @pytest.mark.fast
    def test_exact_divide(self):
        assert nt.ceiling_div(256, 128) == 2

    def test_remainder_rounds_up(self):
        assert nt.ceiling_div(257, 128) == 3
        assert nt.ceiling_div(1, 128) == 1

    def test_zero_dividend(self):
        assert nt.ceiling_div(0, 128) == 0

    def test_one_divisor(self):
        assert nt.ceiling_div(42, 1) == 42


@pytest_marks(["neurotile"])
class TestCeilingDivMisuseGuards:
    def test_b_must_be_positive(self):
        with pytest.raises(AssertionError, match="b must be a positive int"):
            nt.ceiling_div(10, 0)
        with pytest.raises(AssertionError, match="b must be a positive int"):
            nt.ceiling_div(10, -1)


@pytest_marks(["neurotile"])
class TestPTileCount:
    """p_tile_count sizes SBUF P-tile repetitions. A partial trailing P-tile must
    count as ONE tile — a bare floor division returned 0 for a sub-tile extent,
    which propagated a zero-count AP level and crashed alloc with (P, 0)."""

    @pytest.mark.fast
    def test_partial_p_counts_as_one(self):
        # sub-tile extent: the regression case (44 // 128 == 0 before the fix)
        assert p_tile_count(44, 128) == 1
        assert p_tile_count(1, 128) == 1

    def test_exact_and_multiple(self):
        assert p_tile_count(128, 128) == 1
        assert p_tile_count(256, 128) == 2

    def test_partial_after_full_tiles_rounds_up(self):
        # k*tile + partial -> ceil (k+1), so the trailing partial is not dropped
        assert p_tile_count(300, 128) == 3

    def test_nonpositive_tile_p(self):
        assert p_tile_count(44, 0) == 1

    def test_b_must_be_int(self):
        with pytest.raises(AssertionError, match="b must be a positive int"):
            nt.ceiling_div(10, 2.0)
