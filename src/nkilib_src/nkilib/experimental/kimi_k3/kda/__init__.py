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

"""Kimi-K3 Delta Attention kernels and references."""

from .kda_cte import kda_cte
from .kda_cte_torch import kda_cte_chunked_ref, kda_cte_torch_ref
from .kda_decode_batch_nki import kda_decode_batch
from .kda_decode_batch_nki_torch import kda_decode_batch_torch_ref
from .kda_decode_nki import kda_decode_step
from .kda_decode_nki_torch import kda_decode_step_torch_ref
from .kda_prefill_nki import kda_prefill
from .kda_prefill_nki_torch import kda_prefill_torch_ref
from .kda_scan import (
    kda_scan_fori,
    kda_scan_fori_tiled,
    kda_scan_fori_tiled_jit,
)
from .kda_tkg import kda_tkg
from .kda_tkg_torch import kda_tkg_torch_ref

__all__ = [
    "kda_cte",
    "kda_cte_chunked_ref",
    "kda_cte_torch_ref",
    "kda_decode_batch",
    "kda_decode_batch_torch_ref",
    "kda_decode_step",
    "kda_decode_step_torch_ref",
    "kda_prefill",
    "kda_prefill_torch_ref",
    "kda_scan_fori",
    "kda_scan_fori_tiled",
    "kda_scan_fori_tiled_jit",
    "kda_tkg",
    "kda_tkg_torch_ref",
]
