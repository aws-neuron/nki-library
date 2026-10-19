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
"""Kimi-K3 MLA flash attention over an expanded, blocked key/value cache."""

from .mla_flash_attn import BLOCK, MARKER, kimi_k3_mla_flash_attn
from .mla_flash_attn_torch import (
    blocks_to_values,
    keys_to_blocks,
    kimi_k3_mla_flash_attn_torch_ref,
    values_to_blocks,
)

__all__ = [
    "BLOCK",
    "MARKER",
    "blocks_to_values",
    "keys_to_blocks",
    "kimi_k3_mla_flash_attn",
    "kimi_k3_mla_flash_attn_torch_ref",
    "values_to_blocks",
]
