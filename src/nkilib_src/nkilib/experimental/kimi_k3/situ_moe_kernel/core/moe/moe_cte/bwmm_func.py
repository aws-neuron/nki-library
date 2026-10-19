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
"""``BWMMFunc`` enum and the ``_mx_internal_data`` side-channel.

These were previously defined in ``test/integration/.../test_moe_cte_common.py``,
which made them unimportable from ``moe_cte_torch`` (a torch_ref in src/) and
from any non-test consumer. They live in src/ so dispatch works at runtime.

``BWMMFunc`` selects the blockwise-matmul kernel variant.

``_mx_internal_data`` is a module-level dict keyed by
``id(quantization_config)`` that lets the test input generator hand pre-computed
fp32 weights to the MX torch_ref without polluting the kernel input dict.
"""

import weakref
from enum import Enum
from typing import Callable, Tuple

from ....experimental.moe.moe_cte.bwmm_shard_on_block_v2 import (
    bwmm_shard_on_block as bwmm_shard_on_block_v2,
)
from ....experimental.moe.moe_cte.bwmm_shard_on_block_v2 import (
    bwmm_shard_on_block_hybrid,
)
from .bwmm_shard_on_block import bwmm_shard_on_block
from .bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate,
    blockwise_mm_baseline_shard_intermediate_hybrid,
    blockwise_mm_shard_intermediate_dropping,
)


class BWMMFunc(Enum):
    SHARD_ON_BLOCK = 1
    SHARD_ON_HIDDEN = 2
    SHARD_ON_INTERMEDIATE = 3
    SHARD_ON_INTERMEDIATE_AFFINITY_ON_INTERMEDIATE = 4
    SHARD_ON_INTERMEDIATE_HW = 5
    SHARD_ON_INTERMEDIATE_DROPPING = 6
    SHARD_ON_BLOCK_HW = 7
    SHARD_ON_BLOCK_V2 = 8

    MXFP4_SHARD_ON_BLOCK = 10
    MXFP4_SHARD_ON_BLOCK_DYNAMIC = 11

    BASELINE = 100
    BASELINE_ALLOCATED = 200

    def get_bwmm_func(self) -> Tuple[Callable, bool, bool]:
        """Return the kernel function, whether the kernel is nki.jit, whether it is dynamic."""
        if self == BWMMFunc.SHARD_ON_INTERMEDIATE:
            return blockwise_mm_baseline_shard_intermediate, True, False
        elif self == BWMMFunc.SHARD_ON_INTERMEDIATE_HW:
            return blockwise_mm_baseline_shard_intermediate_hybrid, True, True
        elif self == BWMMFunc.SHARD_ON_BLOCK:
            return bwmm_shard_on_block, True, False
        elif self == BWMMFunc.SHARD_ON_BLOCK_V2:
            return bwmm_shard_on_block_v2, True, False
        elif self == BWMMFunc.SHARD_ON_BLOCK_HW:
            return bwmm_shard_on_block_hybrid, True, True
        elif self == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING:
            return blockwise_mm_shard_intermediate_dropping, True, False
        else:
            # NKI's parser does not support `raise`; use `assert` for fatal
            # errors per the compiler's own guidance.
            assert False, f"invalid BWMM kernel name: {self}"  # noqa: S101


# Module-level storage for MX _internal data (keyed by id(spec)).
# Used to pass MX golden computation data to torch ref without polluting
# kernel_input.
_mx_internal_data: dict = {}


def store_mx_internal(spec, data) -> None:
    """Store MX golden ``data`` keyed by ``id(spec)`` with spec-bound lifetime.

    Registers a weakref finalizer so the entry is removed automatically when
    ``spec`` is garbage-collected. This keeps the read API unchanged
    (``_mx_internal_data.get(id(spec))`` while the spec is alive) while
    preventing unbounded accumulation across tests.
    """
    key = id(spec)
    _mx_internal_data[key] = data
    # pop(key, None) is safe even if the key was already removed/reused.
    weakref.finalize(spec, _mx_internal_data.pop, key, None)
