# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import nki.language as nl
from nkilib_src.nkilib.core.utils.common_types import QuantizationType


class KVScaleTest(enum.Enum):
    """KV cache quantization scale for FP8 test configs.

    Use DEFAULT for standard test scale (240/2.3), or pass a float
    literal. Only used when kv_quant=True.
    """

    DEFAULT = "default"


@dataclass
class AttnBlkTestConfig:
    """Test configuration for attention block TKG kernel."""

    batch: int
    q_heads: int
    d_head: int
    H: int
    H_actual: Optional[int]
    S_ctx: int
    S_max_ctx: int
    S_tkg: int

    block_len: int = 0
    update_cache: bool = True
    K_cache_transposed: bool = False
    rmsnorm_X: bool = True
    skip_rope: bool = False
    rope_contiguous_layout: bool = True
    qk_norm_pre_rope: bool = False
    qk_norm_pre_rope_gamma: bool = False
    qk_norm_post_rope: bool = False
    qk_norm_post_rope_gamma: bool = False
    dtype: Any = nl.bfloat16
    quantization_type: QuantizationType = QuantizationType.NONE
    lnc: int = 2
    skip_output_projection: bool = False
    transposed_out: bool = False
    test_bias: bool = False
    input_in_sb: bool = False
    output_in_sb: bool = False
    softmax_scale: Optional[float] = None
    enable_fa_s_prior_tiling: bool = True
    kv_quant: bool = False
    kv_scale: Optional[Union[KVScaleTest, float]] = None
    KVDP: int = 1
    DCP: int = 1
    skip_attention: bool = False

    def __post_init__(self):
        if self.kv_quant and self.kv_scale is None:
            self.kv_scale = KVScaleTest.DEFAULT

    def test_id(self, prefix: str = "") -> str:
        """Generate a unique test ID string from all field values."""
        parts = []
        for val in asdict(self).values():
            if hasattr(val, "value"):
                parts.append(str(val.value))
            else:
                parts.append(str(val))
        if prefix:
            prefix = f"{prefix}_"

        return prefix + "-".join(parts)
