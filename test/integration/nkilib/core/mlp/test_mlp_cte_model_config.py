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
MLP CTE model configuration data

Config format: [vnc_degree, batch, seqlen, hidden, intermediate, tpbSgCyclesSum, norm_type, quant_type,
                fused_add, store_add, skip_gate, act_fn_type, gate_bias, up_bias, down_bias, norm_bias]
"""

from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType

# fmt: off
mlp_cte_model_configs = [
    # [vnc, batch, seqlen, hidden, intermediate, cycles, norm_type, quant_type, fused_add, store_add, skip_gate, act_fn, gate_bias, up_bias, down_bias, norm_bias]
]
# fmt: on
