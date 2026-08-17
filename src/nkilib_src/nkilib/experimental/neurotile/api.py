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


from .core._helpers import ceiling_div, largest_divisor
from .core.factories import alloc_blocks, alloc_tiles, blocks, tiles
from .core.indexing import element_offset
from .core.ndslice import BlockStream, NDSlice  # noqa: F401  -- importable for docs, not in __all__
from .core.psum_pool import psum_pool
from .core.shard_helpers import (
    block_range,
    get_shard_info,
    interleaved_range,
    uneven_block_range,
)

__all__ = [
    # Factories -- views over existing tensors
    "tiles",
    "blocks",
    # Factories -- allocate new buffers
    "alloc_tiles",
    "alloc_blocks",
    "psum_pool",
    # Index markers
    "element_offset",
    # Sharding helpers (each returns a Python slice)
    "block_range",
    "uneven_block_range",
    "interleaved_range",
    "get_shard_info",
    # Trace-time integer math
    "ceiling_div",
    "largest_divisor",
]
