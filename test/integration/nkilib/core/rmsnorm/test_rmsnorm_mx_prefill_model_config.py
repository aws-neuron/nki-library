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
RMSNorm MX prefill model configuration data.

This kernel runs twice per decoder layer in GLM-5.3, in two different configurations, so it
carries two tables rather than one:

  input_layernorm       emits bf16 for attention. Every layer, so 78 of them.
  post_attention norm   fused with the noaux_tc router. The MoE layers only, 75 of them, since
                        the three dense layers (first_k_dense_replace=3) have no router.

Both run on the sequence-parallel shard, so seqlen is the per-rank token count: under CP64 a
128k context puts 2048 tokens on a rank and a 1M context puts 16384.

Router parameters: 256 experts, top_k 8, affinities packed into the row tail so the downstream
bwmm reads them in-kernel rather than via a separate gather. n_group and topk_group are 1, which
makes the group-limiting step a no-op: GLM routes over all 256 experts rather than restricting to
a subset of node groups the way DeepSeek-V3 does with 8 groups and top-4.
"""

from test.utils.common_dataclasses import ModelTestType

# (batch, seqlen, hidden, use_residual, lnc, hidden_interleaved)
NORM_PARAM_NAMES = "batch, seqlen, hidden, use_residual, lnc, hidden_interleaved"

# (batch, seqlen, hidden, expert, n_group, topk_group, top_k, routed_scaling_factor, lnc,
#  pack_affinities)
ROUTER_PARAM_NAMES = (
    "batch, seqlen, hidden, expert, n_group, topk_group, top_k, routed_scaling_factor, lnc, "
    "pack_affinities"
)

_GLM_5_3_NORM_OPTIMAL: list[tuple] = [
    (1, 2048, 6144, True, 2, False),
    (1, 16384, 6144, True, 2, False),
]

_GLM_5_3_ROUTER_OPTIMAL: list[tuple] = [
    (1, 2048, 6144, 256, 1, 1, 8, 2.5, 2, True),
    (1, 16384, 6144, 256, 1, 1, 8, 2.5, 2, True),
]

rmsnorm_mx_prefill_norm_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _GLM_5_3_NORM_OPTIMAL,
    ModelTestType.GENERALITY: [],
}

rmsnorm_mx_prefill_router_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _GLM_5_3_ROUTER_OPTIMAL,
    ModelTestType.GENERALITY: [],
}
