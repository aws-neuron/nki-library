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
MLA QKV CTE model configuration data.

These configs are the test-side twin of the
configuration/test_vector_metadata/test_mla_qkv_cte_*.json files (one per model), which is what
the kernel-to-model benchmark dashboard reads. Keep the two in step: a config in one and not the
other is either registered on the dashboard but never run, or run but never plotted.

The table is spelled out here rather than parsed from that JSON because a model config module is
imported from the installed package layout as well as the source tree, and the metadata directory
is only resolvable in the latter. Parsing it there yields zero configs silently, which surfaces
as a single NOTSET placeholder test rather than an error.

As with the sparse MLA attention CTE kernel, there is no TP x CP x seqlen grid to expand: the
kernel takes an already-sharded per-rank shape, so each entry is one concrete rank shape.

Param format: (vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, kv_lora_rank,
               qk_rope_head_dim, qk_nope_head_dim, norm_eps, precision)

vnc_degree        = logical NeuronCore config; 2 is LNC2, which shards seqlen across 2 cores
seqlen            = queries this rank owns, i.e. the tokens actually projected
hidden            = model hidden dim, 6144 for GLM-5.3
n_heads           = q heads per rank
qk_lora_rank      = Q down-projection rank
kv_lora_rank      = KV latent rank, 512
qk_rope_head_dim  = RoPE half of the per-head Q/K dim
qk_nope_head_dim  = non-RoPE half; 192 for GLM-5.3, 128 for DeepSeek-V3.2
precision         = MlaPrecision member name; BF16 for GLM-5.3

QKV is per-token and has no S_kv axis, so seqlen is the only context-dependent dimension and a
global context S under context parallelism appears here as S / CP.
"""

from test.utils.common_dataclasses import ModelTestType

MODEL_PARAM_NAMES = (
    "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
    "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps, precision"
)

# ``precision`` is the NAME of an MlaPrecision member, not the member itself: this module is
# imported from the installed package layout as well as the source tree, and importing kernel
# code here would couple a config table to it. The test file maps the name back to the enum.
PRECISION_MX = "MX"
PRECISION_BF16 = "BF16"

# GLM-5.3 prefill under the planning doc's TP1 / CP64 attention sharding: 64 q heads per rank,
# hidden 6144, q_lora_rank 2048, kv_lora_rank 512, qk_rope_head_dim 64, qk_nope_head_dim 192.
#
# Registered as OPTIMAL because the model lane runs only the OPTIMAL and TIER0 tiers, so a config
# registered under any other tier is never executed and never reaches the dashboard.
#
# The seqlen ladder is the deployment range expressed per rank at CP64. Two families:
#
#   cold prefill, every token projected:  seqlen = S / 64
#     2048 = 128k,  4096 = 256k,  8192 = 512k,  16384 = 1M
#   warm prefill at 80% prefix-cache hit, only the uncached suffix projected:
#     seqlen = 0.2 x S / 64
#     512 ~= 128k (409.6 rounded up to the tile),  3328 ~= 1M (3276.8 rounded up)
#
# 512 and 3328 are the two operating points the prefill planning doc's roofline table asks for.
# 3328 is 13 whole 256-query tiles, which is what 3276.8 actually costs: the kernel's s-tile is
# 128 rows and LNC2 shards seqlen across 2 cores, so 256 queries is the smallest unit of work
# and a partial tile is paid in full.
_GLM_5_3_OPTIMAL: list[tuple] = [
    # ---- warm prefill, 80% APC. The roofline table's headline cells. ----
    (2, 1, 512, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    (2, 1, 3328, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    # ---- cold prefill, the full deployment context ladder. ----
    (2, 1, 2048, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    (2, 1, 4096, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    (2, 1, 8192, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    (2, 1, 16384, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    # ---- the 256-query floor, where per-token cost stops improving. Tracked so a regression
    # that raises the fixed per-tile cost is visible separately from one that raises the slope.
    (2, 1, 256, 6144, 64, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    # ---- head-sharded variants, in case attention moves off tp=1. ----
    (2, 1, 2048, 6144, 16, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
    (2, 1, 4096, 6144, 8, 2048, 512, 64, 192, 1e-5, PRECISION_BF16),
]

mla_qkv_cte_model_configs: dict[ModelTestType, list[tuple]] = {
    ModelTestType.OPTIMAL: _GLM_5_3_OPTIMAL,
}
