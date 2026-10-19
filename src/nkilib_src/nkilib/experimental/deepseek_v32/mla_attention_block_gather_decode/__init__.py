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
Public API for the DeepSeek-V3.2 MLA decode attention block over indexer-selected keys.

Unlike its sibling packages this one publishes no shape object. It composes two kernels that each
derive their own shape from their own operands, so there is nothing left for the block to derive:
the only extent it needs is ``n_prior``, which it reads off ``gather_indices``.
"""

from .mla_attention_block_gather_decode import mla_attention_block_gather_decode
from .mla_attention_block_gather_decode_torch import mla_attention_block_gather_decode_torch_ref

__all__ = [
    "mla_attention_block_gather_decode",
    "mla_attention_block_gather_decode_torch_ref",
]
