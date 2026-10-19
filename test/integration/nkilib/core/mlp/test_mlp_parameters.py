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

"""Host-side unit tests for pure-Python helpers/constructors in mlp_parameters.

``get_T_from_hidden_input`` and ``MLPParameters.__init__`` only read tensor
metadata (``.buffer``, ``.shape``, ``.dtype``), so they can be exercised on host
with lightweight tensor stubs, without a kernel trace.
"""

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.mlp.mlp_parameters import (
    MLPParameters,
    get_T_from_hidden_input,
)


class _TensorStub:
    """Minimal stand-in exposing only the tensor metadata these helpers read."""

    def __init__(self, shape, buffer=nl.shared_hbm, dtype=nl.bfloat16):
        self.shape = shape
        self.buffer = buffer
        self.dtype = dtype


# ------------------------- get_T_from_hidden_input -------------------------


@pytest.mark.fast
def test_get_T_sbuf_with_scale():
    # SBUF input carrying a scale tensor -> layout [H0, H/512, T]; T is shape[2].
    hidden = _TensorStub((128, 4, 10), buffer=nl.sbuf)
    scale = _TensorStub((128, 1), buffer=nl.sbuf)
    assert get_T_from_hidden_input(hidden, scale) == 10  # ty: ignore[invalid-argument-type]


@pytest.mark.fast
def test_get_T_sbuf_without_scale():
    # SBUF input without scale -> layout [H0, T, H1]; T is shape[1].
    hidden = _TensorStub((128, 7, 4), buffer=nl.sbuf)
    assert get_T_from_hidden_input(hidden, None) == 7  # ty: ignore[invalid-argument-type]


@pytest.mark.fast
def test_get_T_hbm_3d():
    # HBM 3D input [B, S, H] -> T = B * S.
    hidden = _TensorStub((2, 8, 512))
    assert get_T_from_hidden_input(hidden) == 16  # ty: ignore[invalid-argument-type]


@pytest.mark.fast
def test_get_T_hbm_2d():
    # HBM 2D input [T, H] -> T = shape[0].
    hidden = _TensorStub((20, 512))
    assert get_T_from_hidden_input(hidden) == 20  # ty: ignore[invalid-argument-type]


# ------------------------------ MLPParameters ------------------------------


def _weights():
    """Contiguous non-quantized gate/up [H, I] and down [I, H] weight stubs."""
    gate = _TensorStub((512, 512))
    up = _TensorStub((512, 512))
    down = _TensorStub((512, 512))
    return gate, up, down


@pytest.mark.fast
def test_mlpparameters_output_dtype_none_hbm_3d():
    # HBM 3D hidden [B, S, H] path + output_dtype defaulting from hidden dtype.
    gate, up, down = _weights()
    hidden = _TensorStub((1, 4, 512), dtype=nl.bfloat16)
    params = MLPParameters(hidden, gate, up, down, output_dtype=None)  # ty: ignore[invalid-argument-type]
    assert params.output_dtype == nl.bfloat16
    assert params.batch_size == 1
    assert params.sequence_len == 4
    assert params.hidden_size == 512
    assert params.intermediate_size == 512


@pytest.mark.fast
def test_mlpparameters_output_dtype_explicit_hbm_2d():
    # HBM 2D hidden [T, H] path + explicitly supplied output_dtype.
    gate, up, down = _weights()
    hidden = _TensorStub((20, 512))
    params = MLPParameters(hidden, gate, up, down, output_dtype=nl.float32)  # ty: ignore[invalid-argument-type]
    assert params.output_dtype == nl.float32
    assert params.batch_size == 1
    assert params.sequence_len == 20


@pytest.mark.fast
def test_mlpparameters_sbuf_input_no_scale():
    # SBUF hidden [H0, T, H1] without scale.
    gate, up, down = _weights()
    hidden = _TensorStub((128, 6, 4), buffer=nl.sbuf)
    params = MLPParameters(hidden, gate, up, down)  # ty: ignore[invalid-argument-type]
    assert params.input_in_sbuf is True
    assert params.sequence_len == 6


@pytest.mark.fast
def test_mlpparameters_sbuf_input_with_scale():
    # SBUF hidden [H0, H/512, T] with MXFP all-expert scale.
    gate, up, down = _weights()
    hidden = _TensorStub((128, 4, 9), buffer=nl.sbuf)
    scale = _TensorStub((128, 1), buffer=nl.sbuf)
    params = MLPParameters(hidden, gate, up, down, hidden_input_scale=scale)  # ty: ignore[invalid-argument-type]
    assert params.input_in_sbuf is True
    assert params.sequence_len == 9
