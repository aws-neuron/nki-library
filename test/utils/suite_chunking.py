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


def get_chunked_tests(current_chunk_number: int, total_chunk_count: int, tests: list[pytest.Item]) -> list[pytest.Item]:
    if not (1 <= current_chunk_number <= total_chunk_count):
        raise ValueError(
            f"current_chunk_number has to be 1 <= current_chunk_number({current_chunk_number}) <= total_chunk_count({total_chunk_count})"
        )

    # Require at least one test per chunk. When total_chunk_count exceeds the
    # number of collected tests, `base` is 0 and every non-last chunk would be
    # empty, which is almost always a misconfiguration rather than intent.
    if total_chunk_count > len(tests):
        raise ValueError(
            f"total_chunk_count({total_chunk_count}) cannot exceed the number of collected tests({len(tests)}); "
            "this would produce empty chunks"
        )

    start, end = _get_bounds(current_chunk_number - 1, total_chunk_count, len(tests))

    return tests[start:end]


def _get_bounds(curr_index: int, total_chunks: int, full_length: int) -> tuple[int, int]:
    # Balanced chunking: the first `remainder` chunks get one extra test each so
    # that chunk sizes differ by at most 1. The union of all chunks covers every
    # test exactly once, and the slice preserves the collected order (no
    # reordering), so it is a pure function of the input `tests` order.
    base = full_length // total_chunks
    remainder = full_length % total_chunks

    # `min(curr_index, remainder)` accounts for the +1 tests already handed to
    # earlier chunks, keeping the slices contiguous with no gaps or overlaps.
    start = curr_index * base + min(curr_index, remainder)
    end = start + base + (1 if curr_index < remainder else 0)

    return (start, end)
