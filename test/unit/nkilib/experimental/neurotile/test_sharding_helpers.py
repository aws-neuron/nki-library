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

"""Unit tests for slice-based sharding helpers.

`block_range` / `uneven_block_range` / `interleaved_range` are pure
functions that return Python `slice` objects. NDSlice indexing converts
those slices into Grid + Layout primitives; this file pins helper
arithmetic only.

NKI Beta 3 caveats relevant here:
  - `runtime_scalar * 1` is rejected by the parser; the block helper
    must skip the multiply when `owned == 1`.
  - `slice()` constructed in body traces; module-level slice constants
    do not (verified by /tmp/probe_typed_axis/).
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import (
    block_range,
    get_shard_info,
    interleaved_range,
    uneven_block_range,
)

from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# block_range
# ============================================================================


@pytest_marks(["neurotile"])
class TestBlockRange:
    @pytest.mark.fast
    def test_even_division(self):
        s = block_range(rank=0, num_shards=4, total=16)
        assert s == slice(0, 4)

    def test_non_zero_rank(self):
        s = block_range(rank=2, num_shards=4, total=16)
        assert s == slice(8, 12)

    def test_owned_one_collapses_multiply(self):
        """owned == 1 -> start = rank (Beta 3 rejects rt * 1)."""
        s = block_range(rank=3, num_shards=4, total=4)
        assert s == slice(3, 4)

    def test_rejects_non_divisible_total(self):
        with pytest.raises(AssertionError, match="divisible by num_shards"):
            block_range(rank=0, num_shards=3, total=8)

    def test_runtime_rank_owned_one_no_multiply(self):
        """Runtime rank + owned=1: start IS rank (no multiplication)."""

        class _RuntimeScalar:
            def __add__(self, other):
                return ("rt_add", other)

        rt = _RuntimeScalar()
        s = block_range(rank=rt, num_shards=4, total=4)
        # No multiplication happened (we'd see "rt_mul" otherwise).
        assert s.start is rt
        assert s.stop == ("rt_add", 1)

    def test_runtime_rank_owned_gt_one_multiplies(self):
        """Runtime rank + owned > 1: start = rank * owned (runtime expr)."""

        class _MulResult:
            def __add__(self, other):
                return ("rt_add", other)

        class _RuntimeScalar:
            def __init__(self):
                self.mul_called_with = None

            def __mul__(self, other):
                self.mul_called_with = other
                return _MulResult()

        rt = _RuntimeScalar()
        block_range(rank=rt, num_shards=2, total=4)
        # owned = 2 -> rt * 2 was computed
        assert rt.mul_called_with == 2


# ============================================================================
# interleaved_range
# ============================================================================


@pytest_marks(["neurotile"])
class TestInterleavedRange:
    @pytest.mark.fast
    def test_even_division(self):
        s = interleaved_range(rank=0, num_shards=4, total=16)
        assert s == slice(0, 16, 4)

    def test_step_equals_num_shards(self):
        for n in (2, 4, 8):
            s = interleaved_range(rank=0, num_shards=n, total=16)
            assert s.step == n

    def test_start_equals_rank(self):
        s = interleaved_range(rank=2, num_shards=4, total=16)
        assert s.start == 2
        assert s.stop == 16
        assert s.step == 4

    def test_rejects_non_divisible_total(self):
        with pytest.raises(AssertionError, match="divisible by num_shards"):
            interleaved_range(rank=0, num_shards=3, total=10)


# ============================================================================
# uneven_block_range
# ============================================================================


@pytest_marks(["neurotile"])
class TestUnevenBlockRange:
    @pytest.mark.fast
    def test_even_division_matches_block(self):
        s = uneven_block_range(rank=1, num_shards=2, total=8)
        assert s == slice(4, 8)

    def test_uneven_early_ranks_get_extra(self):
        s0 = uneven_block_range(rank=0, num_shards=2, total=7)
        s1 = uneven_block_range(rank=1, num_shards=2, total=7)
        # 7 / 2 -> rank 0 gets 4, rank 1 gets 3.
        assert s0 == slice(0, 4)
        assert s1 == slice(4, 7)

    def test_uneven_remainder_smaller_than_ranks(self):
        # 10 / 4 -> ranks 0,1 get 3 each, ranks 2,3 get 2 each.
        expected = [slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 10)]
        for r in range(4):
            assert uneven_block_range(rank=r, num_shards=4, total=10) == expected[r]

    def test_runtime_rank_requires_even_division(self):
        """Per-rank owned varies in uneven case -> runtime rank impossible."""

        class _RuntimeScalar:
            pass

        rt = _RuntimeScalar()
        with pytest.raises(AssertionError, match="divisible by num_shards"):
            uneven_block_range(rank=rt, num_shards=2, total=7)

    def test_runtime_rank_even_division_works(self):
        """When division is even, runtime rank delegates to block_range."""

        class _MulResult:
            def __add__(self, other):
                return ("rt_add", other)

        class _RuntimeScalar:
            def __init__(self):
                self.mul_called_with = None

            def __mul__(self, other):
                self.mul_called_with = other
                return _MulResult()

        rt = _RuntimeScalar()
        uneven_block_range(rank=rt, num_shards=2, total=8)
        # owned = 4 -> start = rt * 4
        assert rt.mul_called_with == 4


# ============================================================================
# Positive-int validation on shard helpers
# ============================================================================


@pytest_marks(["neurotile"])
class TestShardArgValidation:
    """num_shards and total must be positive ints. Catches off-by-one
    errors and zero-division traps before they manifest as cryptic
    runtime failures."""

    @pytest.mark.fast
    def test_block_range_num_shards_zero_rejected(self):
        with pytest.raises(AssertionError, match="num_shards must be >= 1"):
            block_range(rank=0, num_shards=0, total=8)

    def test_block_range_num_shards_negative_rejected(self):
        with pytest.raises(AssertionError, match="num_shards must be >= 1"):
            block_range(rank=0, num_shards=-2, total=8)

    def test_block_range_total_zero_rejected(self):
        with pytest.raises(AssertionError, match="total must be >= 1"):
            block_range(rank=0, num_shards=2, total=0)

    def test_block_range_num_shards_must_be_int(self):
        with pytest.raises(AssertionError, match="num_shards must be an int"):
            block_range(rank=0, num_shards=2.5, total=8)

    def test_block_range_total_must_be_int(self):
        with pytest.raises(AssertionError, match="total must be an int"):
            block_range(rank=0, num_shards=2, total=8.0)

    def test_interleaved_range_num_shards_zero_rejected(self):
        with pytest.raises(AssertionError, match="num_shards must be >= 1"):
            interleaved_range(rank=0, num_shards=0, total=8)

    def test_uneven_block_range_total_negative_rejected(self):
        with pytest.raises(AssertionError, match="total must be >= 1"):
            uneven_block_range(rank=0, num_shards=2, total=-1)


# ============================================================================
# get_shard_info -- diagnostic helper
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetShardInfo:
    """get_shard_info computes a partition summary for a sharded tile grid."""

    @pytest.mark.fast
    def test_happy_path(self):
        info = get_shard_info(
            tensor_shape=(1024, 4096),
            tile_size=(128, 512),
            shard_dim=0,
            num_shards=2,
            shard_id=0,
        )
        assert info["total_tiles"] == 8  # 1024 // 128
        assert info["tiles_per_shard"] == 4  # 8 // 2
        assert info["shard_id"] == 0
        assert info["num_shards"] == 2
        assert info["shard_dim"] == 0

    def test_happy_path_shard_dim_1(self):
        info = get_shard_info(
            tensor_shape=(1024, 4096),
            tile_size=(128, 512),
            shard_dim=1,
            num_shards=4,
            shard_id=2,
        )
        assert info["total_tiles"] == 8  # 4096 // 512
        assert info["tiles_per_shard"] == 2
        assert info["shard_dim"] == 1

    def test_tensor_shape_must_be_tuple_or_list(self):
        with pytest.raises(AssertionError, match="tensor_shape must be a tuple"):
            get_shard_info(
                tensor_shape=1024,
                tile_size=(128, 512),
                num_shards=2,
                shard_id=0,
            )

    def test_tile_size_must_be_tuple_or_list(self):
        with pytest.raises(AssertionError, match="tile_size must be a tuple"):
            get_shard_info(
                tensor_shape=(1024, 4096),
                tile_size=128,
                num_shards=2,
                shard_id=0,
            )

    def test_rank_mismatch_rejected(self):
        with pytest.raises(AssertionError, match="must have the same"):
            get_shard_info(
                tensor_shape=(1024, 4096, 16),
                tile_size=(128, 512),
                num_shards=2,
                shard_id=0,
            )

    def test_shard_dim_must_be_int(self):
        with pytest.raises(AssertionError, match="shard_dim must be an int"):
            get_shard_info(
                tensor_shape=(1024, 4096),
                tile_size=(128, 512),
                shard_dim="0",
                num_shards=2,
                shard_id=0,
            )

    def test_shard_dim_out_of_range(self):
        with pytest.raises(AssertionError, match="out of range"):
            get_shard_info(
                tensor_shape=(1024, 4096),
                tile_size=(128, 512),
                shard_dim=5,
                num_shards=2,
                shard_id=0,
            )

    def test_shard_dim_negative_rejected(self):
        with pytest.raises(AssertionError, match="out of range"):
            get_shard_info(
                tensor_shape=(1024, 4096),
                tile_size=(128, 512),
                shard_dim=-1,
                num_shards=2,
                shard_id=0,
            )

    def test_non_divisible_rejected(self):
        with pytest.raises(AssertionError, match="not divisible by tile_size"):
            get_shard_info(
                tensor_shape=(1000, 4096),
                tile_size=(128, 512),
                shard_dim=0,
                num_shards=2,
                shard_id=0,
            )
