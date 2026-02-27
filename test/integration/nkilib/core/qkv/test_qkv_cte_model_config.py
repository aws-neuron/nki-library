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
QKV CTE model configuration data.

Config format: [vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,
                norm_type, use_dma_transpose, fused_add, add_bias, norm_bias,
                output_layout, eps]
"""

from nkilib_src.nkilib.core.utils.common_types import NormType, QKVOutputLayout

# Format: [vnc, batch, seqlen, hidden, n_q_heads, n_kv_heads, d_head,
#           norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps]
# fmt: off
qkv_cte_model_configs = [
    # ============ LLAMA3_70B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    [2, 1, 1024, 8192, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=64, CP=1
    [2, 1, 1024, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=16, CP=1
    [2, 1, 1024, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=8,  CP=1
    [2, 1, 1024, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=4,  TP=4,  CP=1
    [2, 1, 512, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=16, TP=8,  CP=2
    [2, 1, 512, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=4,  CP=2
    [2, 1, 256, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=16, CP=4
    [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=8,  CP=8
    [2, 1, 64, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    [2, 1, 10240, 8192, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 10240, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 10240, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 10240, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=4,  TP=4,  CP=1
    [2, 1, 5120, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=8,  CP=2
    [2, 1, 5120, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=4,  CP=2
    [2, 1, 2560, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 1280, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 640, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    [2, 1, 32768, 8192, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 32768, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 32768, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 32768, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=4,  TP=4,  CP=1
    [2, 1, 16384, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=8,  CP=2
    [2, 1, 16384, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=8,  TP=4,  CP=2
    [2, 1, 8192, 8192, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 4096, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 2048, 8192, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=4,  CP=16
    # ============ QWEN3_32B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    [2, 1, 1024, 5120, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=64, CP=1
    [2, 1, 1024, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=16, CP=1
    [2, 1, 1024, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=8,  CP=1
    [2, 1, 1024, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=4,  TP=4,  CP=1
    [2, 1, 512, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=16, TP=8,  CP=2
    [2, 1, 512, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=4,  CP=2
    [2, 1, 256, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=16, CP=4
    [2, 1, 128, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=8,  CP=8
    [2, 1, 64, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    [2, 1, 10240, 5120, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 10240, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 10240, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 10240, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=4,  TP=4,  CP=1
    [2, 1, 5120, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=8,  CP=2
    [2, 1, 5120, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=4,  CP=2
    [2, 1, 2560, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 1280, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 640, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    [2, 1, 32768, 5120, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 32768, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 32768, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 32768, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=4,  TP=4,  CP=1
    [2, 1, 16384, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=8,  CP=2
    [2, 1, 16384, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6], # WS=8,  TP=4,  CP=2
    [2, 1, 8192, 5120, 4, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 4096, 5120, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 2048, 5120, 16, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=4,  CP=16
    # ============ GEMMA3_27B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    [2, 1, 1024, 5376, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=64, CP=1
    [2, 1, 1024, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=16, CP=1
    [2, 1, 1024, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=8,  CP=1
    [2, 1, 1024, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=4,  TP=4,  CP=1
    [2, 1, 512, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=16, TP=8,  CP=2
    [2, 1, 512, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=8,  TP=4,  CP=2
    [2, 1, 256, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=16, CP=4
    [2, 1, 128, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=8,  CP=8
    [2, 1, 64, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],     # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    [2, 1, 10240, 5376, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 10240, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 10240, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 10240, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=4,  TP=4,  CP=1
    [2, 1, 5120, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=16, TP=8,  CP=2
    [2, 1, 5120, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=8,  TP=4,  CP=2
    [2, 1, 2560, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 1280, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 640, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    [2, 1, 32768, 5376, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=64, TP=64, CP=1
    [2, 1, 32768, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=16, CP=1
    [2, 1, 32768, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=8,  CP=1
    [2, 1, 32768, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=4,  TP=4,  CP=1
    [2, 1, 16384, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=16, TP=8,  CP=2
    [2, 1, 16384, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],  # WS=8,  TP=4,  CP=2
    [2, 1, 8192, 5376, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=16, CP=4
    [2, 1, 4096, 5376, 4, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=8,  CP=8
    [2, 1, 2048, 5376, 8, 4, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],   # WS=64, TP=4,  CP=16
]
# fmt: on
