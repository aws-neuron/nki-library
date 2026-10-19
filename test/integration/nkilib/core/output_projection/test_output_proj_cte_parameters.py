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

"""Host-side unit tests for output_projection_cte_parameters helpers.

These exercise pure-Python config/tiling helpers directly (no device, no
simulator) to cover branches the device tests can't reach with runnable
(non-sim-skipped) shapes:

- ``_get_dtype_size`` FP8 branch (returns 1 byte).
- ``_calculate_head_packing`` D>128 fold loop's ``while`` remainder body.
- ``_calculate_double_row_head_packing`` odd-N/odd-D disable path.
- ``build_quantization_config`` "float8e4" sentinel-dtype resolution.
"""

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_cte.output_projection_cte_parameters import (
    _calculate_double_row_head_packing,
    _calculate_head_packing,
    _get_dtype_size,
    build_quantization_config,
)
from nkilib_src.nkilib.core.utils.common_types import DtypeMode, QuantizationType

P_MAX = 128


@pytest.mark.fast
@pytest.mark.parametrize("dtype", [nl.float8_e4m3, nl.float8_e4m3fn])
def test_get_dtype_size_fp8_returns_one(dtype):
    """FP8 dtypes are 1 byte (covers the ``float8_e4m3/fn -> return 1`` arc)."""
    assert _get_dtype_size(dtype) == 1


@pytest.mark.fast
def test_get_dtype_size_known_widths():
    """Sanity on the other width branches (fp32=4, bf16/fp16=2)."""
    assert _get_dtype_size(nl.float32) == 4
    assert _get_dtype_size(nl.bfloat16) == 2
    assert _get_dtype_size(nl.float16) == 2


@pytest.mark.fast
def test_get_dtype_size_default_fallback():
    """A dtype matched by none of the explicit checks hits the final
    ``return 2`` default (covers the fall-through arc)."""
    assert _get_dtype_size(nl.uint8) == 2


@pytest.mark.fast
@pytest.mark.parametrize("d_size", [260, 320])
def test_calculate_head_packing_fold_loop_remainder(d_size):
    """D>128 with div_ceil(D,128) not dividing D exercises the ``while`` fold
    body: for D in {260,320}, div_ceil(.,128)==3 but D%3!=0, so fold_factor is
    incremented until it divides D."""
    n_size, new_d, fold_factor = _calculate_head_packing(2, d_size, P_MAX)
    assert new_d <= P_MAX
    assert d_size % fold_factor == 0
    assert new_d == d_size // fold_factor


@pytest.mark.fast
def test_calculate_double_row_head_packing_disabled_for_odd_n_odd_d():
    """FP8 + h%32==0 + odd N + odd D disables double row (can't split D)."""
    n_size, d_size, use_double_row = _calculate_double_row_head_packing(
        n_size=3, d_size=3, h_size=64, is_fp8_quantized=True
    )
    assert use_double_row is False
    assert (n_size, d_size) == (3, 3)


@pytest.mark.fast
def test_calculate_double_row_head_packing_splits_odd_n_even_d():
    """FP8 + odd N + even D splits D to make N even (contrast branch)."""
    n_size, d_size, use_double_row = _calculate_double_row_head_packing(
        n_size=3, d_size=4, h_size=64, is_fp8_quantized=True
    )
    assert use_double_row is True
    assert (n_size, d_size) == (6, 2)


class _Float8E4SentinelMeta(type):
    """Metaclass so the sentinel *class* stringifies to the 'float8e4' label."""

    def __str__(cls):
        return "float8e4"


class _Float8E4Sentinel(metaclass=_Float8E4SentinelMeta):
    """Opaque weight-dtype label whose str() is the 'float8e4' sentinel.

    Kept a class (not an instance) so it satisfies the ``Optional[type]``
    parameter annotation while still stringifying to the sentinel label.
    """


@pytest.mark.fast
def test_build_quantization_config_float8e4_sentinel_resolves_from_dtype_mode():
    """A 'float8e4' sentinel weight dtype is resolved from dtype_mode (covers
    the sentinel arc), not passed through literally."""
    qc = build_quantization_config(
        quantization_type=QuantizationType.STATIC,
        input_scales=None,
        weight_scales=None,
        input_data_type=None,  # unused on the STATIC path; only read in the MX branch
        weight_data_type=_Float8E4Sentinel,
        dtype_mode=DtypeMode.NON_OCP,
    )
    assert qc.is_enabled
    assert qc.is_fp8_quantized
    assert qc.quant_data_type == nl.float8_e4m3
