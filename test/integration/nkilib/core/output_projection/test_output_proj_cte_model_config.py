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
Output Projection CTE model configuration data.

Config format: [batch, seqlen, hidden, n_head, d_head, test_bias]

Note: For output projection, n_head corresponds to n_q_heads from QKV configs
since output projection operates on the attention output which has n_q_heads.
"""

# Format: [batch, seqlen, hidden, n_head, d_head, test_bias]
# fmt: off
OUTPUT_PROJ_CTE_MODEL_CONFIGS = [
    # ============ LLAMA3_70B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    (1, 1024, 8192, 1, 128, False),   # WS=64, TP=64, CP=1
    (1, 1024, 8192, 4, 128, False),   # WS=16, TP=16, CP=1
    (1, 1024, 8192, 8, 128, False),   # WS=8,  TP=8,  CP=1
    (1, 1024, 8192, 16, 128, False),  # WS=4,  TP=4,  CP=1
    (1, 512, 8192, 8, 128, False),    # WS=16, TP=8,  CP=2
    (1, 512, 8192, 16, 128, False),   # WS=8,  TP=4,  CP=2
    (1, 256, 8192, 4, 128, False),    # WS=64, TP=16, CP=4
    (1, 128, 8192, 8, 128, False),    # WS=64, TP=8,  CP=8
    (1, 64, 8192, 16, 128, False),    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    (1, 10240, 8192, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 10240, 8192, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 10240, 8192, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 10240, 8192, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 5120, 8192, 8, 128, False),   # WS=16, TP=8,  CP=2
    (1, 5120, 8192, 16, 128, False),  # WS=8,  TP=4,  CP=2
    (1, 2560, 8192, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 1280, 8192, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 640, 8192, 16, 128, False),   # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    (1, 32768, 8192, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 32768, 8192, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 32768, 8192, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 32768, 8192, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 16384, 8192, 8, 128, False),  # WS=16, TP=8,  CP=2
    (1, 16384, 8192, 16, 128, False), # WS=8,  TP=4,  CP=2
    (1, 8192, 8192, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 4096, 8192, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 2048, 8192, 16, 128, False),  # WS=64, TP=4,  CP=16
    # ============ QWEN3_32B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    (1, 1024, 5120, 1, 128, False),   # WS=64, TP=64, CP=1
    (1, 1024, 5120, 4, 128, False),   # WS=16, TP=16, CP=1
    (1, 1024, 5120, 8, 128, False),   # WS=8,  TP=8,  CP=1
    (1, 1024, 5120, 16, 128, False),  # WS=4,  TP=4,  CP=1
    (1, 512, 5120, 8, 128, False),    # WS=16, TP=8,  CP=2
    (1, 512, 5120, 16, 128, False),   # WS=8,  TP=4,  CP=2
    (1, 256, 5120, 4, 128, False),    # WS=64, TP=16, CP=4
    (1, 128, 5120, 8, 128, False),    # WS=64, TP=8,  CP=8
    (1, 64, 5120, 16, 128, False),    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    (1, 10240, 5120, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 10240, 5120, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 10240, 5120, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 10240, 5120, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 5120, 5120, 8, 128, False),   # WS=16, TP=8,  CP=2
    (1, 5120, 5120, 16, 128, False),  # WS=8,  TP=4,  CP=2
    (1, 2560, 5120, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 1280, 5120, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 640, 5120, 16, 128, False),   # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    (1, 32768, 5120, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 32768, 5120, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 32768, 5120, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 32768, 5120, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 16384, 5120, 8, 128, False),  # WS=16, TP=8,  CP=2
    (1, 16384, 5120, 16, 128, False), # WS=8,  TP=4,  CP=2
    (1, 8192, 5120, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 4096, 5120, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 2048, 5120, 16, 128, False),  # WS=64, TP=4,  CP=16
    # ============ GEMMA3_27B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    (1, 1024, 5376, 1, 128, False),   # WS=64, TP=64, CP=1
    (1, 1024, 5376, 2, 128, False),   # WS=16, TP=16, CP=1
    (1, 1024, 5376, 4, 128, False),   # WS=8,  TP=8,  CP=1
    (1, 1024, 5376, 8, 128, False),   # WS=4,  TP=4,  CP=1
    (1, 512, 5376, 4, 128, False),    # WS=16, TP=8,  CP=2
    (1, 512, 5376, 8, 128, False),    # WS=8,  TP=4,  CP=2
    (1, 256, 5376, 2, 128, False),    # WS=64, TP=16, CP=4
    (1, 128, 5376, 4, 128, False),    # WS=64, TP=8,  CP=8
    (1, 64, 5376, 8, 128, False),     # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    (1, 10240, 5376, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 10240, 5376, 2, 128, False),  # WS=16, TP=16, CP=1
    (1, 10240, 5376, 4, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 10240, 5376, 8, 128, False),  # WS=4,  TP=4,  CP=1
    (1, 5120, 5376, 4, 128, False),   # WS=16, TP=8,  CP=2
    (1, 5120, 5376, 8, 128, False),   # WS=8,  TP=4,  CP=2
    (1, 2560, 5376, 2, 128, False),   # WS=64, TP=16, CP=4
    (1, 1280, 5376, 4, 128, False),   # WS=64, TP=8,  CP=8
    (1, 640, 5376, 8, 128, False),    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    (1, 32768, 5376, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 32768, 5376, 2, 128, False),  # WS=16, TP=16, CP=1
    (1, 32768, 5376, 4, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 32768, 5376, 8, 128, False),  # WS=4,  TP=4,  CP=1
    (1, 16384, 5376, 4, 128, False),  # WS=16, TP=8,  CP=2
    (1, 16384, 5376, 8, 128, False),  # WS=8,  TP=4,  CP=2
    (1, 8192, 5376, 2, 128, False),   # WS=64, TP=16, CP=4
    (1, 4096, 5376, 4, 128, False),   # WS=64, TP=8,  CP=8
    (1, 2048, 5376, 8, 128, False),   # WS=64, TP=4,  CP=16
    # ============ GPT_OSS CONFIGS ============
    # --- Original Sequence Length 1024 ---
    (1, 1024, 3072, 1, 64, True),    # WS=64, TP=64, CP=1
    (1, 1024, 3072, 4, 64, True),    # WS=16, TP=16, CP=1
    (1, 1024, 3072, 8, 64, True),    # WS=8,  TP=8,  CP=1
    (1, 1024, 3072, 16, 64, True),   # WS=4,  TP=4,  CP=1
    (1, 512, 3072, 8, 64, True),     # WS=16, TP=8,  CP=2
    (1, 512, 3072, 16, 64, True),    # WS=8,  TP=4,  CP=2
    (1, 256, 3072, 4, 64, True),     # WS=64, TP=16, CP=4
    (1, 128, 3072, 8, 64, True),     # WS=64, TP=8,  CP=8
    (1, 64, 3072, 16, 64, True),     # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    (1, 10240, 3072, 1, 64, True),   # WS=64, TP=64, CP=1
    (1, 10240, 3072, 4, 64, True),   # WS=16, TP=16, CP=1
    (1, 10240, 3072, 8, 64, True),   # WS=8,  TP=8,  CP=1
    (1, 10240, 3072, 16, 64, True),  # WS=4,  TP=4,  CP=1
    (1, 5120, 3072, 8, 64, True),    # WS=16, TP=8,  CP=2
    (1, 5120, 3072, 16, 64, True),   # WS=8,  TP=4,  CP=2
    (1, 2560, 3072, 4, 64, True),    # WS=64, TP=16, CP=4
    (1, 1280, 3072, 8, 64, True),    # WS=64, TP=8,  CP=8
    (1, 640, 3072, 16, 64, True),    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    (1, 32768, 3072, 1, 64, True),   # WS=64, TP=64, CP=1
    (1, 32768, 3072, 4, 64, True),   # WS=16, TP=16, CP=1
    (1, 32768, 3072, 8, 64, True),   # WS=8,  TP=8,  CP=1
    (1, 32768, 3072, 16, 64, True),  # WS=4,  TP=4,  CP=1
    (1, 16384, 3072, 8, 64, True),   # WS=16, TP=8,  CP=2
    (1, 16384, 3072, 16, 64, True),  # WS=8,  TP=4,  CP=2
    (1, 8192, 3072, 4, 64, True),    # WS=64, TP=16, CP=4
    (1, 4096, 3072, 8, 64, True),    # WS=64, TP=8,  CP=8
    (1, 2048, 3072, 16, 64, True),   # WS=64, TP=4,  CP=16
    # ============ Qwen3-235B CONFIGS ============
    # --- Original Sequence Length 1024 ---
    (1, 1024, 4096, 1, 128, False),   # WS=64, TP=64, CP=1
    (1, 1024, 4096, 4, 128, False),   # WS=16, TP=16, CP=1
    (1, 1024, 4096, 8, 128, False),   # WS=8,  TP=8,  CP=1
    (1, 1024, 4096, 16, 128, False),  # WS=4,  TP=4,  CP=1
    (1, 512, 4096, 8, 128, False),    # WS=16, TP=8,  CP=2
    (1, 512, 4096, 16, 128, False),   # WS=8,  TP=4,  CP=2
    (1, 256, 4096, 4, 128, False),    # WS=64, TP=16, CP=4
    (1, 128, 4096, 8, 128, False),    # WS=64, TP=8,  CP=8
    (1, 64, 4096, 16, 128, False),    # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 10240 ---
    (1, 10240, 4096, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 10240, 4096, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 10240, 4096, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 10240, 4096, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 5120, 4096, 8, 128, False),   # WS=16, TP=8,  CP=2
    (1, 5120, 4096, 16, 128, False),  # WS=8,  TP=4,  CP=2
    (1, 2560, 4096, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 1280, 4096, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 640, 4096, 16, 128, False),   # WS=64, TP=4,  CP=16
    # --- Original Sequence Length 32768 ---
    (1, 32768, 4096, 1, 128, False),  # WS=64, TP=64, CP=1
    (1, 32768, 4096, 4, 128, False),  # WS=16, TP=16, CP=1
    (1, 32768, 4096, 8, 128, False),  # WS=8,  TP=8,  CP=1
    (1, 32768, 4096, 16, 128, False), # WS=4,  TP=4,  CP=1
    (1, 16384, 4096, 8, 128, False),  # WS=16, TP=8,  CP=2
    (1, 16384, 4096, 16, 128, False), # WS=8,  TP=4,  CP=2
    (1, 8192, 4096, 4, 128, False),   # WS=64, TP=16, CP=4
    (1, 4096, 4096, 8, 128, False),   # WS=64, TP=8,  CP=8
    (1, 2048, 4096, 16, 128, False),  # WS=64, TP=4,  CP=16
]
# fmt: on
