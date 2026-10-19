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

"""GPT-OSS decode sampling-tail megakernel (experimental)."""

from .tail_chunked_megakernel import compose_hops, gpt_oss_tail_chunked_megakernel, owned_row_block, owned_rows
from .tail_chunked_megakernel_torch import gpt_oss_tail_chunked_ref, rank_row_block

__all__ = [
    "compose_hops",
    "gpt_oss_tail_chunked_megakernel",
    "gpt_oss_tail_chunked_ref",
    "owned_row_block",
    "owned_rows",
    "rank_row_block",
]
