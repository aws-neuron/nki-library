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
"""AttnRes --- Kimi-K3's cross-layer residual mixture kernel.

NEW code, not a fork: a softmax mixture over the current residual stream plus the
snapshots banked at earlier layers. Replaces the torch ``apply_attn_res`` (a whole-bank
``cat`` plus a per-token bmm) which dominated the compiled graph's instruction count.
See ``attn_res_cte_torch.py`` for the reference algebra.
"""

from .attn_res_cte import attn_res_cte
from .attn_res_cte_torch import attn_res_cte_torch_ref

__all__ = [
    "attn_res_cte",
    "attn_res_cte_torch_ref",
]
