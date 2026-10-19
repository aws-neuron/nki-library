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

"""FlashAttention-2 forward for Trainium 3."""

from .config import FlashAttentionConfig, FlashAttentionGeometry
from .flash_attention_2 import flash_attention_2
from .flash_attention_fwd import FlashAttentionFwd
from .remainder_stream import RemainderBlockStream, block_table, remainder_block_stream

__all__ = [
    "FlashAttentionConfig",
    "FlashAttentionFwd",
    "FlashAttentionGeometry",
    "RemainderBlockStream",
    "block_table",
    "flash_attention_2",
    "remainder_block_stream",
]
