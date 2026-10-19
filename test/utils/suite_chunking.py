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
from typing import TypeVar

# The chunking is index-only: it strides the collected list without inspecting the
# items, so it preserves whatever item type the caller collected.
_Test = TypeVar("_Test")


def get_chunked_tests(current_chunk_number: int, total_chunk_count: int, tests: list[_Test]) -> list[_Test]:
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

    # Stride rather than slice: collection order groups tests by file and
    # parametrization, which correlates with cost, so a contiguous slice
    # concentrates the expensive ones (one suite: 6.5 vs 150.5 min at 2 chunks).
    # Sizes still differ by at most 1, and each chunk keeps collection order.
    return tests[current_chunk_number - 1 :: total_chunk_count]
