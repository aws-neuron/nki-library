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
import pytest

from test.utils.suite_chunking import get_chunked_tests


def _collect_all_chunks(total_chunk_count: int, tests: list) -> list[list]:
    """Return the chunk each of 1..total_chunk_count would produce for `tests`."""
    return [
        get_chunked_tests(current_chunk_number=n, total_chunk_count=total_chunk_count, tests=tests)
        for n in range(1, total_chunk_count + 1)
    ]


class TestGetChunkedTests:
    @pytest.mark.parametrize(
        "num_tests, total_chunk_count",
        [
            (10, 3),
            (6, 3),
            (9, 3),
            (100, 7),
            (5, 5),
            (4, 3),
            (7, 7),
            (20, 6),
            (3, 3),
            (1, 1),
        ],
    )
    def test_union_of_chunks_covers_every_test_exactly_once(self, num_tests, total_chunk_count):
        tests = list(range(num_tests))

        chunks = _collect_all_chunks(total_chunk_count, tests)

        # Concatenating chunks in order reproduces the original list: no dropped,
        # duplicated, or reordered tests.
        assert [test for chunk in chunks for test in chunk] == tests

    @pytest.mark.parametrize(
        "num_tests, total_chunk_count",
        [
            (10, 3),
            (100, 7),
            (20, 6),
            (5, 5),
            (4, 3),
        ],
    )
    def test_chunks_are_balanced_within_one(self, num_tests, total_chunk_count):
        tests = list(range(num_tests))

        chunks = _collect_all_chunks(total_chunk_count, tests)

        sizes = [len(chunk) for chunk in chunks]
        assert max(sizes) - min(sizes) <= 1

    def test_remainder_goes_to_leading_chunks(self):
        # 10 tests / 3 chunks -> [4, 3, 3]: the first `remainder` chunks get +1.
        chunks = _collect_all_chunks(3, list(range(10)))

        assert [len(chunk) for chunk in chunks] == [4, 3, 3]

    def test_no_chunk_is_empty_for_valid_config(self):
        chunks = _collect_all_chunks(5, list(range(5)))

        assert all(len(chunk) >= 1 for chunk in chunks)

    def test_slice_preserves_collection_order(self):
        # Determinism prerequisite for pytest-xdist: chunking never reorders.
        tests = ["c", "a", "b", "z", "y", "x"]

        chunks = _collect_all_chunks(3, tests)

        assert [test for chunk in chunks for test in chunk] == tests

    @pytest.mark.parametrize("current_chunk_number", [0, -1, 4])
    def test_chunk_number_out_of_range_raises(self, current_chunk_number):
        with pytest.raises(ValueError, match="current_chunk_number"):
            get_chunked_tests(
                current_chunk_number=current_chunk_number,
                total_chunk_count=3,
                tests=list(range(10)),
            )

    @pytest.mark.parametrize(
        "num_tests, total_chunk_count",
        [
            (2, 5),
            (0, 3),
            (4, 5),
        ],
    )
    def test_more_chunks_than_tests_raises(self, num_tests, total_chunk_count):
        with pytest.raises(ValueError, match="empty chunks"):
            get_chunked_tests(
                current_chunk_number=1,
                total_chunk_count=total_chunk_count,
                tests=list(range(num_tests)),
            )
