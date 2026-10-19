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

import os as _os

if _os.environ.get("SKIP_MODEL_TESTS"):
    raise ImportError("Model tests skipped via SKIP_MODEL_TESTS")

"""
MLA V-up + o_proj CTE model configuration data.

The test-side twin of configuration/test_vector_metadata/test_mla_vup_oproj_cte_glm_5_3.json,
which is what the kernel-to-model benchmark dashboard reads. A config in one and not the other
is either registered on the dashboard but never run, or run but never plotted.

Param format: (batch, seqlen, hidden, L, D_V, HID)

seqlen  = tokens this rank owns
hidden  = q heads per rank
L       = kv_lora_rank, 512
D_V     = v_head_dim, 256 for GLM-5.3
HID     = o_proj output width, the model hidden size

Only the bf16 path is registered. GLM-5.3 runs V-up and o_proj in bf16, so the MX variant of
this kernel (which additionally takes compact_scales) is not a GLM configuration.
"""

from test.utils.common_dataclasses import ModelTestType

MODEL_PARAM_NAMES = "batch, seqlen, hidden, L, D_V, HID"

# GLM-5.3, 64 q heads, v_head_dim 256, o_proj out 6144. Registered as OPTIMAL because the model
# lane runs only the OPTIMAL and TIER0 tiers.
#
# The first two entries are the deployment cells the dashboard aggregates: under CP64 a 128k
# context puts 2048 tokens on each rank and a 1M context puts 16384. Those have to match the
# shapes the aggregation derives, or the cell resolves to no data and the whole row is dropped
# for having a missing kernel. The rest are TP shards, kept for coverage breadth.
_GLM_5_3_OPTIMAL: list[tuple] = [
    # deployment cells: 128k and 1M global context at CP64, full 64 heads on one rank
    (1, 2048, 64, 512, 256, 6144),
    (1, 16384, 64, 512, 256, 6144),
    # TP shards: 32 heads (TP2), 16 (TP4), 8 (TP8)
    (1, 256, 32, 512, 256, 6144),
    (1, 128, 16, 512, 256, 6144),
    (1, 256, 8, 512, 256, 6144),
    # short per-rank token counts at the full head count
    (1, 128, 64, 512, 256, 6144),
    (1, 256, 64, 512, 256, 6144),
]

mla_vup_oproj_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _GLM_5_3_OPTIMAL,
    ModelTestType.GENERALITY: [],
}
