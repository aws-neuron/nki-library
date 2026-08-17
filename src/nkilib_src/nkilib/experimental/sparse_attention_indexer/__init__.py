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

"""DeepSeek Sparse Attention Indexer kernel module.

Entry kernel:
  * ``sparse_attention_indexer_mx_bf16score`` — MX (block-32 fp8) projections
    with a BF16 score matmul (no Hadamard, no Q/K quant); K cache is bf16.
    Emits per-query top-k position indices via hardware ``nisa.topk``.

Returns ``(index_score_hbm, topk_idx_hbm)``.
"""

from .sparse_attention_indexer_mx_bf16score import (
    sparse_attention_indexer_mx_bf16score,
)
from .sparse_attention_indexer_utils import SAIConfig

__all__ = [
    "SAIConfig",
    "sparse_attention_indexer_mx_bf16score",
]
