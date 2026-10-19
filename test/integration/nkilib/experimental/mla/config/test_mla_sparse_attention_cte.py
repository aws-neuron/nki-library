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
Sparse MLA attention CTE model configuration data.

These configs are the test-side twin of the
configuration/test_vector_metadata/test_mla_sparse_attention_cte_*.json files (one per model,
DeepSeek-V3.2 and GLM-5.3), which is what the kernel-to-model benchmark dashboard reads. Keep
the two in step: a config in one and not the other is either registered on the dashboard but
never run, or run but never plotted.

The table is spelled out here rather than parsed from that JSON because a model config module
is imported from the installed package layout as well as the source tree, and the metadata
directory is only resolvable in the latter. Parsing it there yields zero configs silently,
which surfaces as a single NOTSET placeholder test rather than an error.

Unlike the other model config modules, there is no TP x CP x seqlen grid to expand: the
kernel takes an already-sharded per-rank shape, so each entry is one concrete rank shape.

Param format: (batch, seqlen, S_kv, hidden, K, L, R, dense, precision)

seqlen        = queries this rank owns
S_kv          = KV length this rank attends over
hidden        = q heads per rank
K             = index_topk (ignored when dense)
L             = kv_lora_rank, 512 for both DeepSeek-V3.2 and GLM-5.3
R             = qk_rope_head_dim
dense         = dense causal mask instead of the sparse top-k path
precision     = MlaPrecision member name: MX for DeepSeek-V3.2, BF16 for GLM-5.3

The attention kernel itself reads bf16 latents on both paths, so precision selects the
arithmetic of the surrounding projections, not the dtype of this kernel's inputs.
"""

from test.utils.common_dataclasses import ModelTestType

MODEL_PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R, dense, precision"

# ``precision`` is the NAME of an MlaPrecision member, not the member itself: this module is
# imported from the installed package layout as well as the source tree, and importing kernel
# code here would couple a config table to it. The test file maps the name back to the enum.
# It is a config field rather than a fixed default because the two models run different paths
# and the same shape can appear in both, so a measurement is only identified with it.
PRECISION_MX = "MX"
PRECISION_BF16 = "BF16"

# DeepSeek-V3.2 prefill, 128 q heads per rank, kv_lora_rank 512, qk_rope_head_dim 64.
# Registered as OPTIMAL because the model lane runs only the OPTIMAL and TIER0 tiers, so a
# config registered under any other tier is never executed and never reaches the dashboard.
_DEEPSEEK_V32_OPTIMAL: list[tuple] = [
    # sparse flow, full DeepSeek dims
    (1, 128, 8192, 128, 2048, 512, 64, False, PRECISION_MX),
    # sparse flow, small per-rank query counts at full DeepSeek dims
    (1, 2, 8192, 128, 2048, 512, 64, False, PRECISION_MX),
    (1, 16, 8192, 128, 2048, 512, 64, False, PRECISION_MX),
    (1, 32, 8192, 128, 2048, 512, 64, False, PRECISION_MX),
    # Dense flow
    (1, 128, 1024, 128, 16, 512, 64, True, PRECISION_MX),
    (1, 128, 2048, 128, 16, 512, 64, True, PRECISION_MX),
]

# GLM-5.3 prefill on the bf16 MLA path, 64 q heads per rank. Shares kv_lora_rank 512,
# qk_rope_head_dim 64 and index_topk 2048 with DeepSeek-V3.2, so heads per rank and the
# arithmetic path are the only fields that move. Shapes mirror the deployment cases in
# BF16_SPARSE_PARAMS / BF16_DENSE_PARAMS rather than a generated grid.
_GLM_5_3_OPTIMAL: list[tuple] = [
    # sparse flow, base deployment shape
    (1, 128, 8192, 64, 2048, 512, 64, False, PRECISION_BF16),
    # sparse flow, shorter gathered caches
    (1, 256, 4096, 64, 2048, 512, 64, False, PRECISION_BF16),
    (1, 128, 2048, 64, 2048, 512, 64, False, PRECISION_BF16),
    # TP shards: 32 heads per rank (TP2) and 16 (TP4, the floor H % 16 allows)
    (1, 256, 2048, 32, 512, 512, 64, False, PRECISION_BF16),
    (1, 256, 2048, 16, 512, 512, 64, False, PRECISION_BF16),
    # S_global 128k, the CP deployment target, reachable only via the DMA-indirect gather
    (1, 1024, 131072, 64, 2048, 512, 64, False, PRECISION_BF16),
    (1, 2048, 131072, 64, 2048, 512, 64, False, PRECISION_BF16),
    (1, 16384, 1048576, 64, 2048, 512, 64, False, PRECISION_BF16),
    # dense flow (S <= index_topk, indexer skipped), the range production always takes dense
    (1, 32, 128, 64, 16, 512, 64, True, PRECISION_BF16),
    (1, 128, 128, 64, 16, 512, 64, True, PRECISION_BF16),
]

mla_sparse_attention_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _DEEPSEEK_V32_OPTIMAL + _GLM_5_3_OPTIMAL,
    ModelTestType.GENERALITY: [],
}
