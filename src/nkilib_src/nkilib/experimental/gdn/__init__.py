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

"""Gated DeltaNet (GDN) kernels for Qwen3.5 hybrid linear attention.

Prefill (chunked gated delta-rule) and decode (single-token recurrence), plus
the fused decode megakernel (in_proj + conv1d + GQA + recurrence in one @nki.jit).
"""

from .gdn_block_tkg import gdn_block_tkg
from .gdn_conv1d import gdn_conv1d_decode, gdn_conv1d_prefill
from .gdn_cte import gdn_cte
from .gdn_tkg import gdn_tkg

__all__ = [
    "gdn_block_tkg",
    "gdn_conv1d_decode",
    "gdn_conv1d_prefill",
    "gdn_cte",
    "gdn_tkg",
]
