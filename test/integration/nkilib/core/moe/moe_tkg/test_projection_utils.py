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

"""Host-side unit tests for projection_utils.py.

convert_params_to_views is pure Python (attribute rebinds plus the pure
convert_to_view()/is_quant() helpers), so it is exercised directly with stub
tensors, without a device or simulator.
"""

from typing import cast

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.moe.moe_tkg.mlp_parameters import MLPParameters
from nkilib_src.nkilib.core.moe.moe_tkg.projection_utils import convert_params_to_views
from nkilib_src.nkilib.core.utils.common_types import QuantizationType


class _Stub:
    """Minimal stand-in for an nl.NkiTensor exposing shape/dtype/buffer."""

    def __init__(self, shape, dtype=nl.bfloat16, buffer="hbm"):
        self.shape = shape
        self.dtype = dtype
        self.buffer = buffer


def StubTensor(shape, dtype=nl.bfloat16, buffer="hbm") -> nl.NkiTensor:
    """nl.NkiTensor stand-in, cast so the type checker accepts it (runtime-identical)."""
    return cast(nl.NkiTensor, _Stub(shape, dtype, buffer))


def _fp32(shape):
    return StubTensor(shape, dtype=nl.float32)


def _params(quant_type=QuantizationType.NONE):
    H, I = 256, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate, up, down = StubTensor((H, I)), StubTensor((H, I)), StubTensor((I, H))
    scales = {}
    if quant_type == QuantizationType.STATIC:
        scales = {
            "gate_w_scale": _fp32((128, 1)),
            "up_w_scale": _fp32((128, 1)),
            "down_w_scale": _fp32((128, 1)),
            "gate_up_in_scale": _fp32((128, 1)),
            "down_in_scale": _fp32((128, 1)),
        }
    return MLPParameters(hidden, gate, up, down, quantization_type=quant_type, **scales)


@pytest.mark.fast
def test_convert_params_to_views_quantized_with_gate():
    # gate weights present (enters the guarded rebind) and is_quant() True
    # (quant scales are also converted).
    p = _params(QuantizationType.STATIC)
    convert_params_to_views(p)


@pytest.mark.fast
def test_convert_params_to_views_unquantized_without_gate():
    # gate weights absent (guard skipped) and is_quant() False (quant convert skipped).
    p = _params(QuantizationType.NONE)
    p.gate_proj_weights_tensor = None
    convert_params_to_views(p)
