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
# SPDX-License-Identifier: Apache-2.0
"""Standalone pure-CPU GPT-OSS MXFP4 decode golden (kernel-development oracle)."""

from .config import GptOssConfig
from .golden_entry import build_golden_model, gptoss_mxfp4_decode
from .layer_entry import dequantize_expert_weights, gptoss_mxfp4_decode_layer
from .model import GptOssForCausalLM, GptOssModel
from .paged_kv import PagedKVManager

__all__ = [
    "GptOssConfig",
    "GptOssForCausalLM",
    "GptOssModel",
    "PagedKVManager",
    "build_golden_model",
    "gptoss_mxfp4_decode",
    "gptoss_mxfp4_decode_layer",
    "dequantize_expert_weights",
]
