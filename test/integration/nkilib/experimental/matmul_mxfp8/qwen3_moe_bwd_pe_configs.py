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

"""Qwen3-235B TP1 MoE backward 1x32 PE phase configs.

Shapes are the per-core LNC2 matmuls executed by the fused kernel.
"""

from typing import NamedTuple


class MoeBwdPePhaseConfig(NamedTuple):
    M: int
    K: int
    N: int
    tile_m: int
    tile_k: int
    tile_n: int
    block_m: int
    block_n: int
    block_k: int
    load_m: int
    load_n: int
    spill_reload: bool


QWEN3_235B_TP1_MOE_BWD_1X32_PE_CONFIGS = {
    "phase1": MoeBwdPePhaseConfig(4096, 4096, 768, 128, 512, 256, 8, 3, 8, 8, 3, False),
    "phase2": MoeBwdPePhaseConfig(4096, 3072, 2048, 128, 512, 512, 8, 4, 3, 4, 4, False),
    "phase3": MoeBwdPePhaseConfig(3072, 4096, 2048, 128, 512, 512, 24, 2, 2, 24, 2, False),
    "phase4": MoeBwdPePhaseConfig(2048, 4096, 1536, 128, 512, 512, 16, 3, 2, 16, 3, False),
}
