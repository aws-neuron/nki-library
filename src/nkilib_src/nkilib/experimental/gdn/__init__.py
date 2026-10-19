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

"""Gated DeltaNet (GDN) kernels for hybrid linear attention.

Prefill (chunked gated delta-rule) and decode (single-token recurrence), plus the fused
decode megakernel (in_proj + conv1d + GQA + recurrence in one @nki.jit).

PREFILL has ONE entry point, `gdn_cte`, with two candidate intra-chunk inverse algorithms:
pass `algorithm=ALGORITHM_SCHUR` (the default, and the faster of the two) or
`ALGORITHM_NEUMANN`. The implementations are neither exported nor `@nki.jit`-decorated --
they are traced bodies `gdn_cte` inlines. Head dims may be anything up to `MAX_HEAD_DIM`
(128, the partition bound), with q and k matching. Selecting neumann restricts the call
(no GQA head sharing, no supplied `recurrent_state`, Dv must equal Dk, `mask_pack`
unused): it uses a flat [B, S, D] layout with one head dim for q/k/v, and a zero initial
state. See README.md for the interface table and the comparison.
"""

from .gdn_block_tkg import gdn_block_tkg
from .gdn_conv1d import gdn_conv1d_decode, gdn_conv1d_prefill
from .gdn_cte import gdn_cte
from .gdn_cte_utils import (
    ALGORITHM_NEUMANN,
    ALGORITHM_SCHUR,
    CHUNK,
    DEFAULT_ALGORITHM,
    MAX_HEAD_DIM,
    N_MASKS,
    build_mask_pack,
)
from .gdn_tkg import gdn_tkg

__all__ = [
    "ALGORITHM_NEUMANN",
    "ALGORITHM_SCHUR",
    "CHUNK",
    "DEFAULT_ALGORITHM",
    "MAX_HEAD_DIM",
    "N_MASKS",
    "build_mask_pack",
    "gdn_block_tkg",
    "gdn_conv1d_decode",
    "gdn_conv1d_prefill",
    "gdn_cte",
    "gdn_tkg",
]
