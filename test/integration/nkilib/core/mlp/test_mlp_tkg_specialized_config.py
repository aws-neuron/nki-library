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

"""Host-side unit tests for ``_is_llama3_70b_specialized_config``.

The function short-circuits to False as soon as any field diverges from the
Llama3-70B static-FP8 base config. Each test builds a base-matching params stub
then flips exactly one field to exercise that field's early-return arc. All these
arcs execute before the ``get_verified_program_sharding_info`` call (which needs a
backend/trace), so they are host-testable.
"""

from types import SimpleNamespace

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg.mlp_tkg import _is_llama3_70b_specialized_config
from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType


def _base():
    """params stub whose fields all match the Llama3-70B specialization base."""
    return SimpleNamespace(
        hidden_size=8192,
        intermediate_size=3584,
        output_dtype=nl.bfloat16,
        quant_params=SimpleNamespace(quantization_type=QuantizationType.STATIC),
        norm_params=SimpleNamespace(
            normalization_type=NormType.RMS_NORM,
            normalization_bias_tensor=None,
        ),
        activation_fn=ActFnType.SiLU,
        fused_add_params=SimpleNamespace(fused_add_tensor=None, store_fused_add_result=False),
        bias_params=SimpleNamespace(
            gate_proj_bias_tensor=None,
            up_proj_bias_tensor=None,
            down_proj_bias_tensor=None,
        ),
        skip_gate_proj=False,
        store_output_in_sbuf=False,
        input_in_sbuf=False,
        transposed_in=False,
        transposed_out=False,
        use_tkg_down_proj_optimized_layout=False,
    )


@pytest.mark.fast
def test_specialized_config_hidden_size_mismatch():
    p = _base()
    p.hidden_size = 4096
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_intermediate_size_mismatch():
    p = _base()
    p.intermediate_size = 4096
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_output_dtype_mismatch():
    p = _base()
    p.output_dtype = nl.float32
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_quantization_type_mismatch():
    p = _base()
    p.quant_params.quantization_type = QuantizationType.NONE
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_normalization_type_mismatch():
    p = _base()
    p.norm_params.normalization_type = NormType.LAYER_NORM
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_activation_fn_mismatch():
    p = _base()
    p.activation_fn = ActFnType.GELU
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_has_fused_add():
    p = _base()
    p.fused_add_params.fused_add_tensor = object()
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_store_fused_add():
    p = _base()
    p.fused_add_params.store_fused_add_result = True
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_has_projection_bias():
    p = _base()
    p.bias_params.gate_proj_bias_tensor = object()
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_has_normalization_bias():
    p = _base()
    p.norm_params.normalization_bias_tensor = object()
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_skip_gate_proj():
    p = _base()
    p.skip_gate_proj = True
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_store_output_in_sbuf():
    p = _base()
    p.store_output_in_sbuf = True
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_input_in_sbuf():
    p = _base()
    p.input_in_sbuf = True
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_transposed():
    p = _base()
    p.transposed_in = True
    assert _is_llama3_70b_specialized_config(p) is False


@pytest.mark.fast
def test_specialized_config_optimized_down_layout():
    p = _base()
    p.use_tkg_down_proj_optimized_layout = True
    assert _is_llama3_70b_specialized_config(p) is False
