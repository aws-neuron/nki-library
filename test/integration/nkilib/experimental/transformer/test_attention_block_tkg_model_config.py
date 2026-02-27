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
Attention Block TKG model configuration data

Config format: [batch, num_heads, d_head, H, H_actual, S_ctx, S_max_ctx, S_tkg, block_len,
                update_cache, K_cache_transposed, rmsnorm_X, skip_rope, rope_contiguous_layout,
                rmsnorm_QK, qk_norm_post_rope, qk_norm_post_rope_gamma, dtype, quantization_type,
                lnc, transposed_out, test_bias, input_in_sb, output_in_sb, softmax_scale, kv_quant]
"""

import nki.language as nl
from nkilib_src.nkilib.core.utils.common_types import QuantizationType

# fmt: off
attention_block_tkg_model_configs = [
    # batch, num_heads, d_head, H, H_actual, S_ctx, S_max_ctx, S_tkg, block_len, update_cache, K_cache_transposed, rmsnorm_X, skip_rope, rope_contiguous_layout, rmsnorm_QK, qk_norm_post_rope, qk_norm_post_rope_gamma, dtype, quantization_type, lnc, transposed_out, test_bias, input_in_sb, output_in_sb, softmax_scale, kv_quant
]
# fmt: on
