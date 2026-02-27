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

Config format: [AttnTKGConfig, AttnTKGTestParams]
"""

from nkilib_src.nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
from test.integration.nkilib.core.attention.test_attention_tkg_utils import AttnTKGTestParams

attention_tkg_model_configs = [
    # ============ QWEN3_32B CONFIGS ============
    [AttnTKGConfig(128, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(256, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1024, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(512, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(256, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(512, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(128, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1024, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1024, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(512, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(128, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(256, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1024, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(128, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(512, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(256, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(64, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(32, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(16, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(64, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(32, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(16, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(16, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(64, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(32, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(32, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(64, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(16, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(1, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2048, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2048, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2048, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(2048, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4096, 1, 1, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4096, 1, 1, 1024, 1024, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4096, 1, 1, 131072, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4096, 1, 1, 32768, 32768, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
]
