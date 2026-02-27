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
MoE CTE model configuration data

Config format: [bwmm_func_enum, hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, bias,
                training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower,
                up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I]

Where bwmm_func_enum is a BWMMFunc enum value (e.g., BWMMFunc.SHARD_ON_INTERMEDIATE)
"""

# fmt: off
# Model configs will be populated here

moe_cte_model_configs = [
]
# fmt: on
