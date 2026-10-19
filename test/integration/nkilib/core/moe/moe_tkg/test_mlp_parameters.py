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

"""Host-side unit tests for mlp_parameters.py.

These exercise the pure-Python parameter construction and validation logic
(quantization / normalization / bias dispatch, input-layout resolution, and
argument validation) directly, without a device or simulator. They run in
milliseconds and are identical on every platform.
"""

from typing import cast

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe.moe_tkg import mlp_parameters as mp
from nkilib_src.nkilib.core.moe.moe_tkg.mlp_parameters import (
    MLPBiasParameters,
    MLPFusedAddParameters,
    MLPNormalizationParameters,
    MLPParameters,
    MLPQuantizationParameters,
    _validate_mlp_arguments_restrictions,
    _validate_mlp_arguments_shapes,
    get_T_from_hidden_input,
)
from nkilib_src.nkilib.core.utils.common_types import (
    ComputationMode,
    NormType,
    QuantizationType,
)


class _Stub:
    """Minimal stand-in for an nl.NkiTensor exposing shape/dtype/buffer."""

    def __init__(self, shape, dtype=nl.bfloat16, buffer="hbm"):
        self.shape = shape
        self.dtype = dtype
        self.buffer = buffer


def StubTensor(shape, dtype=nl.bfloat16, buffer="hbm") -> nl.NkiTensor:
    """Build a lightweight nl.NkiTensor stand-in.

    The validation/config code under test only reads .shape/.dtype/.buffer, so a
    tiny stub suffices. It is cast to nl.NkiTensor (runtime-identical) so the type
    checker accepts it where a real tensor is expected — the same idiom used for
    the test doubles in test/unit/test_ucode_lib.py.
    """
    return cast(nl.NkiTensor, _Stub(shape, dtype, buffer))


class _WeightsStub:
    """Holds the two weight tensors read by MLPQuantizationParameters._validate_shapes."""

    def __init__(self, hidden, intermediate):
        # _validate_shapes reads shape[-1] of each: down -> H, up -> I.
        self.down_proj_weights_tensor = StubTensor((intermediate, hidden))
        self.up_proj_weights_tensor = StubTensor((hidden, intermediate))


def _fp32(shape):
    return StubTensor(shape, dtype=nl.float32)


def _u8(shape):
    return StubTensor(shape, dtype=nl.uint8)


# ---------------------------------------------------------------------------
# get_T_from_hidden_input
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_get_T_prequantized_input_uses_last_dim():
    # hidden_input_scale present -> T is the last dimension.
    hidden = StubTensor((4, 3, 16))
    assert get_T_from_hidden_input(hidden, hidden_input_scale=StubTensor((1,))) == 16


@pytest.mark.fast
def test_get_T_sbuf_input_uses_dim1():
    hidden = StubTensor((4, 8, 16), buffer=nl.sbuf)
    assert get_T_from_hidden_input(hidden) == 8


@pytest.mark.fast
def test_get_T_hbm_3d_input_multiplies_leading_dims():
    # HBM 3D [B, S, H] -> T = B * S.
    hidden = StubTensor((4, 8, 16), buffer="hbm")
    assert get_T_from_hidden_input(hidden) == 32


@pytest.mark.fast
def test_get_T_hbm_2d_input_uses_dim0():
    hidden = StubTensor((7, 16), buffer="hbm")
    assert get_T_from_hidden_input(hidden) == 7


# ---------------------------------------------------------------------------
# MLPQuantizationParameters._validate_dtype
# ---------------------------------------------------------------------------


def _quant_params(quant_type, gate_w=None, up_w=None, down_w=None, gate_up_in=None, down_in=None):
    return MLPQuantizationParameters(
        quantization_type=quant_type,
        gate_w_scale=gate_w,
        up_w_scale=up_w,
        down_w_scale=down_w,
        gate_up_in_scale=gate_up_in,
        down_in_scale=down_in,
        clipping_bound=0.0,
    )


@pytest.mark.fast
def test_validate_dtype_none_takes_no_scale_branches():
    # NONE -> the ROW/STATIC group, STATIC group and MX group are all skipped.
    _quant_params(QuantizationType.NONE)._validate_dtype()


@pytest.mark.fast
def test_validate_dtype_row_requires_fp32_weight_scales():
    q = _quant_params(QuantizationType.ROW, gate_w=_fp32((128, 4)), up_w=_fp32((128, 4)), down_w=_fp32((128, 8)))
    q._validate_dtype()


@pytest.mark.fast
def test_validate_dtype_static_requires_fp32_weight_and_input_scales():
    q = _quant_params(
        QuantizationType.STATIC,
        gate_w=_fp32((128, 1)),
        up_w=_fp32((128, 1)),
        down_w=_fp32((128, 1)),
        gate_up_in=_fp32((128, 1)),
        down_in=_fp32((128, 1)),
    )
    q._validate_dtype()


@pytest.mark.fast
def test_validate_dtype_mx_requires_uint8_weight_scales():
    q = _quant_params(QuantizationType.MX, gate_w=_u8((128, 4)), up_w=_u8((128, 4)), down_w=_u8((128, 8)))
    q._validate_dtype()


# ---------------------------------------------------------------------------
# MLPQuantizationParameters._validate_shapes
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_validate_shapes_static():
    H, I = 256, 512
    q = _quant_params(
        QuantizationType.STATIC,
        gate_w=_fp32((128, 1)),
        up_w=_fp32((128, 1)),
        down_w=_fp32((128, 1)),
        gate_up_in=_fp32((128, 1)),
        down_in=_fp32((128, 1)),
    )
    q._validate_shapes(_WeightsStub(H, I))


@pytest.mark.fast
def test_validate_shapes_row():
    H, I = 256, 512
    q = _quant_params(
        QuantizationType.ROW,
        gate_w=_fp32((128, I)),
        up_w=_fp32((128, I)),
        down_w=_fp32((128, H)),
    )
    q._validate_shapes(_WeightsStub(H, I))


@pytest.mark.fast
def test_validate_shapes_row_mx():
    # ROW_MX physical scale layout: gate/up [_PMAX, n_I512*_Q_WIDTH], down [_PMAX, H//_PMAX].
    H, I = 512, 512
    n_I512 = -(-I // (128 * 4))  # ceil
    gate_up_cols = n_I512 * 4
    down_cols = H // 128
    q = _quant_params(
        QuantizationType.ROW_MX,
        gate_w=_fp32((128, gate_up_cols)),
        up_w=_fp32((128, gate_up_cols)),
        down_w=_fp32((128, down_cols)),
    )
    q._validate_shapes(_WeightsStub(H, I))


# ---------------------------------------------------------------------------
# convert_to_view methods (quant / fused-add / bias)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_quant_convert_to_view_with_all_scales_present():
    q = _quant_params(
        QuantizationType.STATIC,
        gate_w=_fp32((128, 1)),
        up_w=_fp32((128, 1)),
        down_w=_fp32((128, 1)),
        gate_up_in=_fp32((128, 1)),
        down_in=_fp32((128, 1)),
    )
    q.convert_to_view()


@pytest.mark.fast
def test_quant_convert_to_view_with_no_scales():
    _quant_params(QuantizationType.NONE).convert_to_view()


# ---------------------------------------------------------------------------
# MLPFusedAddParameters
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_fused_add_validate_dtype_and_convert_with_tensor():
    fa = MLPFusedAddParameters(fused_add_tensor=StubTensor((8, 256), dtype=nl.bfloat16), store_fused_add_result=True)
    fa._validate_dtype()
    fa.convert_to_view()


@pytest.mark.fast
def test_fused_add_validate_dtype_and_convert_without_tensor():
    fa = MLPFusedAddParameters(fused_add_tensor=None, store_fused_add_result=False)
    fa._validate_dtype()
    fa.convert_to_view()


# ---------------------------------------------------------------------------
# MLPNormalizationParameters
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_normalization_no_norm_clears_tensors():
    n = MLPNormalizationParameters(
        NormType.NO_NORM,
        normalization_weights_tensor=StubTensor((1, 256)),
        normalization_bias_tensor=StubTensor((1, 256)),
    )
    assert n.normalization_weights_tensor is None
    assert n.normalization_bias_tensor is None


@pytest.mark.fast
def test_normalization_rms_keeps_tensors_and_validates_dtype():
    n = MLPNormalizationParameters(
        NormType.RMS_NORM,
        normalization_weights_tensor=StubTensor((1, 256), dtype=nl.bfloat16),
        normalization_bias_tensor=StubTensor((1, 256), dtype=nl.bfloat16),
    )
    n._validate_dtype()


@pytest.mark.fast
def test_normalization_validate_dtype_without_tensors():
    # weights/bias absent -> both dtype checks are skipped.
    n = MLPNormalizationParameters(
        NormType.RMS_NORM,
        normalization_weights_tensor=None,
        normalization_bias_tensor=None,
    )
    n._validate_dtype()


# ---------------------------------------------------------------------------
# MLPBiasParameters
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_bias_validate_dtype_and_convert_with_all_tensors():
    b = MLPBiasParameters(
        gate_proj_bias_tensor=StubTensor((1, 512), dtype=nl.bfloat16),
        up_proj_bias_tensor=StubTensor((1, 512), dtype=nl.bfloat16),
        down_proj_bias_tensor=StubTensor((1, 256), dtype=nl.bfloat16),
    )
    b._validate_dtype()
    b.convert_to_view()


@pytest.mark.fast
def test_bias_validate_dtype_and_convert_without_tensors():
    # No bias tensors -> every per-tensor dtype/convert branch is skipped.
    b = MLPBiasParameters(None, None, None)
    b._validate_dtype()
    b.convert_to_view()


# ---------------------------------------------------------------------------
# MLPParameters.__init__ input-layout resolution
# ---------------------------------------------------------------------------


def _weights(hidden, intermediate):
    return StubTensor((hidden, intermediate)), StubTensor((hidden, intermediate)), StubTensor((intermediate, hidden))


@pytest.mark.fast
def test_init_transposed_in_4d_input():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    # Transposed input shape: [H0, n_prgs, H1_shard, BxS].
    hidden = StubTensor((2, 1, 128, 8), buffer="hbm")
    p = MLPParameters(hidden, gate, up, down, transposed_in=True)
    assert p.sequence_len == 8


@pytest.mark.fast
def test_init_sbuf_input_with_prequant_scale():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((4, 3, 8), buffer=nl.sbuf)  # [H0, H/512, T]
    scale = StubTensor((1,))
    p = MLPParameters(hidden, gate, up, down, hidden_input_scale=scale)
    assert p.sequence_len == 8


@pytest.mark.fast
def test_init_sbuf_input_without_scale():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((4, 8, 3), buffer=nl.sbuf)  # [H0, T, H1], no scale
    p = MLPParameters(hidden, gate, up, down)
    assert p.sequence_len == 8


@pytest.mark.fast
def test_init_hbm_prequant_scale():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((4, 3, 8), buffer="hbm")  # [H0, H/512, T] in HBM
    scale = StubTensor((1,))
    p = MLPParameters(hidden, gate, up, down, hidden_input_scale=scale)
    assert p.sequence_len == 8


@pytest.mark.fast
def test_init_hbm_3d_bsh_input():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((2, 4, H), buffer="hbm")  # [B, S, H]
    p = MLPParameters(hidden, gate, up, down)
    assert p.batch_size == 2 and p.sequence_len == 4


@pytest.mark.fast
def test_init_output_dtype_defaults_to_input_dtype():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((8, H), buffer="hbm", dtype=nl.bfloat16)
    p = MLPParameters(hidden, gate, up, down, output_dtype=None)
    assert p.output_dtype == nl.bfloat16


@pytest.mark.fast
def test_init_output_dtype_explicit_override():
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((8, H), buffer="hbm", dtype=nl.bfloat16)
    out_dtype = np.dtype(np.float32)
    p = MLPParameters(hidden, gate, up, down, output_dtype=out_dtype)
    assert p.output_dtype == out_dtype


@pytest.mark.fast
def test_init_down_weights_3d_moe_layout():
    H, I = 256, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate, up = StubTensor((H, I)), StubTensor((H, I))
    down = StubTensor((3, I, H))  # [E, I, H]
    p = MLPParameters(hidden, gate, up, down)
    assert p.intermediate_size == I


@pytest.mark.fast
def test_init_down_weights_2d_dense_layout():
    H, I = 256, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate, up, down = _weights(H, I)  # down is [I, H]
    p = MLPParameters(hidden, gate, up, down)
    assert p.intermediate_size == I


# ---------------------------------------------------------------------------
# _validate_mlp_arguments_shapes (MX dtype/quant checks + projection biases)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_validate_shapes_mx_dtype_and_quant_alignment():
    H, I = 512, 512
    hidden = StubTensor((8, H), buffer="hbm")
    # MX weights are 3D-packed: gate/up [_pmax, H/512, I], down [_pmax, I/512, H].
    gate = StubTensor((128, 1, I))
    up = StubTensor((128, 1, I))
    down = StubTensor((128, 1, H))
    p = MLPParameters(
        hidden,
        gate,
        up,
        down,
        quantization_type=QuantizationType.MX,
        gate_w_scale=_u8((128, 4)),
        up_w_scale=_u8((128, 4)),
        down_w_scale=_u8((128, 8)),
    )
    _validate_mlp_arguments_shapes(p)


@pytest.mark.fast
def test_validate_shapes_row_mx_dtype_only_no_intermediate_check():
    # ROW_MX is an MX dtype (H%512 checked) but not is_quant_mx (no I alignment check).
    H, I = 512, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate = StubTensor((128, 1, I))
    up = StubTensor((128, 1, I))
    down = StubTensor((128, 1, H))
    n_I512 = -(-I // (128 * 4))  # ceil
    p = MLPParameters(
        hidden,
        gate,
        up,
        down,
        quantization_type=QuantizationType.ROW_MX,
        gate_w_scale=_fp32((128, n_I512 * 4)),
        up_w_scale=_fp32((128, n_I512 * 4)),
        down_w_scale=_fp32((128, H // 128)),
    )
    _validate_mlp_arguments_shapes(p)


@pytest.mark.fast
def test_validate_shapes_with_projection_biases():
    H, I = 256, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate, up, down = _weights(H, I)
    p = MLPParameters(
        hidden,
        gate,
        up,
        down,
        gate_proj_bias_tensor=StubTensor((1, I), dtype=nl.bfloat16),
        up_proj_bias_tensor=StubTensor((1, I), dtype=nl.bfloat16),
        down_proj_bias_tensor=StubTensor((1, H), dtype=nl.bfloat16),
    )
    _validate_mlp_arguments_shapes(p)


# ---------------------------------------------------------------------------
# _validate_mlp_arguments_restrictions (TKG vs CTE + TKG-only flags)
#
# nl.program_ndim() requires a device context, so patch it to a valid value
# (0 == no specialization) for these host-only tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_program_ndim(monkeypatch):
    monkeypatch.setattr(mp.nl, "program_ndim", lambda: 0)


def _tkg_params(**kwargs):
    # Small T (<= threshold) selects TKG mode.
    H, I = 256, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate, up, down = _weights(H, I)
    return MLPParameters(hidden, gate, up, down, mode=ComputationMode.DECODE, **kwargs)


@pytest.mark.fast
def test_restrictions_tkg_default(patched_program_ndim):
    _validate_mlp_arguments_restrictions(_tkg_params())


@pytest.mark.fast
def test_restrictions_tkg_optimized_down_layout(patched_program_ndim):
    _validate_mlp_arguments_restrictions(
        _tkg_params(use_tkg_down_proj_optimized_layout=True, use_tkg_down_proj_column_tiling=False)
    )


@pytest.mark.fast
def test_restrictions_tkg_transposed_out(patched_program_ndim):
    _validate_mlp_arguments_restrictions(_tkg_params(transposed_out=True))


@pytest.mark.fast
def test_restrictions_tkg_mx_quant(patched_program_ndim):
    H, I = 512, 512
    hidden = StubTensor((8, H), buffer="hbm")
    gate = StubTensor((128, 1, I))
    up = StubTensor((128, 1, I))
    down = StubTensor((128, 1, H))
    p = MLPParameters(
        hidden,
        gate,
        up,
        down,
        mode=ComputationMode.DECODE,
        quantization_type=QuantizationType.MX,
        gate_w_scale=_u8((128, 4)),
        up_w_scale=_u8((128, 4)),
        down_w_scale=_u8((128, 8)),
    )
    _validate_mlp_arguments_restrictions(p)


@pytest.mark.fast
def test_restrictions_tkg_transposed_in(patched_program_ndim):
    # transposed_in (HBM 4D) exercises the input_in_sbuf-or-transposed_in and
    # transposed_in restriction branches in TKG mode.
    H, I = 256, 512
    gate, up, down = _weights(H, I)
    hidden = StubTensor((2, 1, 128, 8), buffer="hbm")  # [H0, n_prgs, H1_shard, BxS]
    p = MLPParameters(hidden, gate, up, down, mode=ComputationMode.DECODE, transposed_in=True)
    _validate_mlp_arguments_restrictions(p)


@pytest.mark.fast
def test_restrictions_cte_mode(patched_program_ndim):
    # Large T selects CTE mode (the else branch of is_tkg).
    H, I = 256, 512
    hidden = StubTensor((256, H), buffer="hbm")
    gate, up, down = _weights(H, I)
    p = MLPParameters(hidden, gate, up, down, mode=ComputationMode.PREFILL)
    _validate_mlp_arguments_restrictions(p)
