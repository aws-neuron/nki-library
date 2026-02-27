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
Attention TKG model configuration data

Config format: [batch, q_head, s_active, s_prior, s_prior_full, d_head, block_len,
                tp_k_prior, use_pos_id, fuse_rope, outsInSB, qkvInSB, test_sink, strided_mm1, dtype]
"""

import nki.language as nl

attention_tkg_model_configs = [
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype
    # ============ QWEN3_32B CONFIGS ============
    [1024, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1024, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1024, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1024, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [128, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [128, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [128, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [128, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [16, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [16, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [16, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [16, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [1, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2048, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2048, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2048, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2048, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [256, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [256, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [256, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [256, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [2, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [32, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [32, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [32, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [32, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4096, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4096, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4096, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4096, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [4, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [512, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [512, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [512, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [512, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [64, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [64, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [64, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [64, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [8, 1, 1, 10240, 10240, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [8, 1, 1, 1024, 1024, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [8, 1, 1, 131072, 131072, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
    [8, 1, 1, 32768, 32768, 128, 0, True, True, True, False, False, False, True, nl.bfloat16, False],
]
