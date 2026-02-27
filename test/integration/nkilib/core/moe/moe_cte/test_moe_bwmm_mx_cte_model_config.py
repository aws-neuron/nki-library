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
MoE BWMM MX CTE model configuration data for GPT OSS

All configs use: H=3072, I=3072, vnc_degree=2 (LNC2)

Config format: [vnc_degree, hidden, tokens, intermediate(I_TP), expert, block_size, top_k,
                act_fn, expert_affinities_scaling_mode, dtype, weight_dtype,
                skip_mode, bias, is_dynamic, gate_clamp_upper, gate_clamp_lower,
                up_clamp_upper, up_clamp_lower, use_uint_weights]

Kernel constraints (shard-on-block MX):
  - B % 128 == 0
  - 512 <= H <= 8192, H % 512 == 0
  - I_TP % 512 == 0 OR (I_TP < 512 AND I_TP % 32 == 0)
  - num_shards == 2
"""

import nki.language as nl

from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# fmt: off
moe_bwmm_mx_cte_model_configs = [
    # ============ EP1TP8: E=128, TP=8, I_TP=384 (3072/8) ============
    # --- Sequence Length 1024 ---
    [2, 3072, 1024, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP4
    [2, 3072, 1024, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP8 e4m3
    [2, 3072, 1024, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP4
    [2, 3072, 1024, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP8 e4m3
    [2, 3072, 1024, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP8 e5m2
    # --- Sequence Length 4096 ---
    [2, 3072, 4096, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP4
    [2, 3072, 4096, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP8 e4m3
    [2, 3072, 4096, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP4
    [2, 3072, 4096, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP8 e4m3
    # --- Sequence Length 10240 ---
    [2, 3072, 10240, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP4
    [2, 3072, 10240, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP8 e4m3
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP4
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP8 e4m3
    # --- Sequence Length 32768 ---
    [2, 3072, 32768, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP4
    [2, 3072, 32768, 384, 128, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=128, static, skip, MXFP8 e4m3
    [2, 3072, 32768, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP4
    [2, 3072, 32768, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],  # EP1TP8, B=256, static, skip, MXFP8 e4m3
    # ============ EP4TP2: E=32, TP=2, I_TP=1536 (3072/2) ============
    # --- Sequence Length 1024 ---
    [2, 3072, 1024, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 1024, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 1024, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 1024, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 4096 ---
    [2, 3072, 4096, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 4096, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 4096, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 4096, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 10240 ---
    [2, 3072, 10240, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 10240, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 10240, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 10240, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 32768 ---
    [2, 3072, 32768, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 32768, 1536, 32, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 32768, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 32768, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP4TP2, B=256, dynamic, skip, MXFP8 e5m2
    # ============ EP32TP2: E=4, TP=2, I_TP=1536 (3072/2) ============
    # --- Sequence Length 1024 ---
    [2, 3072, 1024, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 1024, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 1024, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 1024, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 4096 ---
    [2, 3072, 4096, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 4096, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 4096, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 4096, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 10240 ---
    [2, 3072, 10240, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 10240, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 10240, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 10240, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 32768 ---
    [2, 3072, 32768, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP4
    [2, 3072, 32768, 1536, 4, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 32768, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP4
    [2, 3072, 32768, 1536, 4, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP32TP2, B=256, dynamic, skip, MXFP8 e5m2
    # ============ EP8TP8: E=16, TP=8, I_TP=384 (3072/8) ============
    # --- Sequence Length 1024 ---
    [2, 3072, 1024, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP4
    [2, 3072, 1024, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP4
    [2, 3072, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 4096 ---
    [2, 3072, 4096, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP4
    [2, 3072, 4096, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 4096, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP4
    [2, 3072, 4096, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 10240 ---
    [2, 3072, 10240, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP4
    [2, 3072, 10240, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 10240, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP4
    [2, 3072, 10240, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP8 e5m2
    # --- Sequence Length 32768 ---
    [2, 3072, 32768, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP4
    [2, 3072, 32768, 384, 16, 128, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=128, dynamic, skip, MXFP8 e5m2
    [2, 3072, 32768, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP4
    [2, 3072, 32768, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],  # EP8TP8, B=256, dynamic, skip, MXFP8 e5m2
]
# fmt: on
