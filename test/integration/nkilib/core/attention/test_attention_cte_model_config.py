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
"""
Attention CTE model configuration data

Config format: [bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len, cp_degree, cp_rank_id, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink]
"""

# fmt: off
attention_cte_model_configs = [
    # [bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len, cp_degree, cp_rank_id, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink]
]
# fmt: on
