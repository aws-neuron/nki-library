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
"""Model shapes used by QKV tests.

Kept separate from the generated model configs so that tests parametrized on their
own model list can read these shapes even where model configs are unavailable.
"""

from typing import NotRequired, TypedDict


class QkvModelSpec(TypedDict):
    n_q_heads: int
    n_kv_heads: int
    d_head: int
    hidden: int
    bias: bool
    use_gamma: NotRequired[bool]


MODELS: dict[str, QkvModelSpec] = {
    "llama3_70b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "hidden": 8192, "bias": False},
    "qwen3_32b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "hidden": 5120, "bias": False},
    "qwen3_235b": {"n_q_heads": 64, "n_kv_heads": 4, "d_head": 128, "hidden": 4096, "bias": False, "use_gamma": True},
    "gemma3_27b": {"n_q_heads": 32, "n_kv_heads": 16, "d_head": 128, "hidden": 5376, "bias": False},
    "gptoss_120b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 64, "hidden": 3072, "bias": True},
}
