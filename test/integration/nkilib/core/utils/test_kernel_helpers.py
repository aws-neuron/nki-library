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
"""Simulation-mode branch-coverage tests for core/utils/kernel_helpers.py.

Two flavors of test live here:

* Pure-Python helpers (config / shape / dtype math) are called directly and
  validated with plain assertions. These need no backend.
* Helpers that touch ``nl.ndarray`` / ``nki.isa`` / ``nisa.get_nc_version()``
  are exercised inside a small ``@nki.jit`` kernel run through ``nki.simulate``
  (no ``trace_only`` marker, so they contribute coverage in simulation mode).
  The ``@nki.jit`` + ``nki.simulate`` trace path always targets gen4
  (``nisa.get_nc_version()`` returns gen4 regardless of ``NKI_NC_VERSION``), so no
  per-test hardware-generation setup is needed.
"""

from unittest.mock import Mock

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils import kernel_helpers as kh
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import ActFnType, DtypeMode
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert

from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata

# =============================================================================
# Pure-Python helpers (no backend required)
# =============================================================================


@pytest_test_metadata(name="KernelHelpers PSUM Banks")
@pytest_marks(["kernel_helpers"])
class TestPsumNumBanks:
    """_psum_num_banks: 8 banks normally, 7 when dma_transpose lowers to PE transpose."""

    @pytest.mark.fast
    def test_all_banks_usable_by_default(self, monkeypatch):
        monkeypatch.setattr(kh, "_NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", False)
        assert kh._psum_num_banks() == 8

    @pytest.mark.fast
    def test_reserves_bank_for_pe_transpose(self, monkeypatch):
        monkeypatch.setattr(kh, "_NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", True)
        assert kh._psum_num_banks() == 7


@pytest_marks(["kernel_helpers"])
class TestGetNlActFnFromType:
    """get_nl_act_fn_from_type: enum -> nl.* activation function for every case."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "act_fn,expected",
        [
            (ActFnType.SiLU, nl.silu),
            (ActFnType.GELU, nl.gelu),
            (ActFnType.GELU_Tanh_Approx, nl.gelu_apprx_tanh),
            (ActFnType.Swish, nl.gelu_apprx_sigmoid),
            (ActFnType.ReLU, nl.relu),
        ],
    )
    def test_maps_each_activation(self, act_fn, expected):
        assert kh.get_nl_act_fn_from_type(act_fn) == expected


@pytest_marks(["kernel_helpers"])
class TestGetMaxPositiveValueForDtype:
    """get_max_positive_value_for_dtype: FP8 max magnitudes, None for others."""

    @pytest.mark.fast
    def test_float8e4_string_alias(self):
        # "float8e4" string is remapped to nl.float8_e4m3 before the lookup.
        assert kh.get_max_positive_value_for_dtype("float8e4") == 240.0

    @pytest.mark.fast
    def test_e4m3_dtype(self):
        assert kh.get_max_positive_value_for_dtype(nl.float8_e4m3) == 240.0

    @pytest.mark.fast
    def test_e5m2_dtype(self):
        assert kh.get_max_positive_value_for_dtype(nl.float8_e5m2) == 57344.0

    @pytest.mark.fast
    def test_unmapped_dtype_returns_none(self):
        assert kh.get_max_positive_value_for_dtype(nl.float32) is None


@pytest_marks(["kernel_helpers"])
class TestReduce:
    """reduce: fold a list with mul / add / min / max starting from initial_value."""

    @pytest.mark.fast
    def test_mul(self):
        assert kh.reduce("mul", [2, 3, 4], 1) == 24

    @pytest.mark.fast
    def test_add(self):
        assert kh.reduce("add", [2, 3, 4], 0) == 9

    @pytest.mark.fast
    def test_min(self):
        assert kh.reduce("min", [5, 2, 9], 100) == 2

    @pytest.mark.fast
    def test_max(self):
        assert kh.reduce("max", [5, 2, 9], 0) == 9


@pytest_marks(["kernel_helpers"])
class TestResolveFp8E4m3Dtype:
    """resolve_fp8_e4m3_dtype: OCP / NON_OCP resolve host-side; AUTO needs trace."""

    @pytest.mark.fast
    def test_ocp(self):
        assert kh.resolve_fp8_e4m3_dtype(DtypeMode.OCP) == nl.float8_e4m3fn

    @pytest.mark.fast
    def test_non_ocp(self):
        assert kh.resolve_fp8_e4m3_dtype(DtypeMode.NON_OCP) == nl.float8_e4m3


@pytest_marks(["kernel_helpers"])
class TestResolveDtypeToNki:
    """resolve_dtype_to_nki: string dtype -> nl dtype for every recognized case.

    The 'float8_e4m3fn' arc is covered separately because it needs a trace-time
    ``nisa.get_nc_version()`` (see test_resolve_dtype_to_nki_float8_e4m3fn).
    """

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("bool", nl.bool_),
            ("int8", nl.int8),
            ("int16", nl.int16),
            ("int32", nl.int32),
            ("uint8", nl.uint8),
            ("uint16", nl.uint16),
            ("uint32", nl.uint32),
            ("float16", nl.float16),
            ("float32", nl.float32),
            ("bfloat16", nl.bfloat16),
            ("float8_e4m3", nl.float8_e4m3),
            ("float8e4", nl.float8_e4m3),
            ("float8_e5m2", nl.float8_e5m2),
            ("float8e5", nl.float8_e5m2),
            (str(nl.float4_e2m1fn_x4), nl.float4_e2m1fn_x4),
            (str(nl.float8_e4m3fn_x4), nl.float8_e4m3fn_x4),
            (str(nl.float8_e5m2_x4), nl.float8_e5m2_x4),
        ],
    )
    def test_recognized_dtype_strings(self, dtype, expected):
        assert kh.resolve_dtype_to_nki(dtype) == expected


# =============================================================================
# Backend / device helpers (run through nki.simulate)
# =============================================================================


# Distinct sentinels so a test can assert WHICH fp8 dtype the resolver returned,
# not merely that the kernel traced. The @nki.jit + nki.simulate trace path always
# targets gen4 (nisa.get_nc_version() == gen4 regardless of NKI_NC_VERSION), so the
# gen4-branch dtype, float8_e4m3fn, is the expected result here.
_FP8_E4M3FN_SENTINEL = 1.0
_FP8_E4M3_SENTINEL = 2.0


def _emit_fp8_sentinel(out, resolved):
    """Write a sentinel identifying the resolved fp8 dtype into the HBM output."""
    sentinel = _FP8_E4M3FN_SENTINEL if resolved == nl.float8_e4m3fn else _FP8_E4M3_SENTINEL
    sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sb, value=sentinel)
    nisa.dma_copy(dst=out, src=sb)


@nki.jit
def _resolve_dtype_kernel(dtype_str: str):
    """Exercise resolve_dtype_to_nki for a string that needs trace context, emitting
    a sentinel identifying which nki dtype it resolved to."""
    resolved = kh.resolve_dtype_to_nki(dtype_str)
    scratch = nl.ndarray((1, 4), dtype=resolved, buffer=nl.sbuf)
    nisa.memset(dst=scratch, value=0)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    _emit_fp8_sentinel(out, resolved)
    return out


@nki.jit
def _resolve_fp8_auto_kernel():
    """Exercise resolve_fp8_e4m3_dtype(AUTO), which calls nisa.get_nc_version(),
    emitting a sentinel identifying which nki dtype it resolved to."""
    resolved = kh.resolve_fp8_e4m3_dtype(DtypeMode.AUTO)
    scratch = nl.ndarray((1, 4), dtype=resolved, buffer=nl.sbuf)
    nisa.memset(dst=scratch, value=0)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    _emit_fp8_sentinel(out, resolved)
    return out


@pytest.mark.fast
def test_resolve_dtype_to_nki_float8_e4m3fn():
    """resolve_dtype_to_nki('float8_e4m3fn') resolves to float8_e4m3fn under the
    gen4 simulate trace (its nisa.get_nc_version() == gen4 branch)."""
    result = nki.simulate(_resolve_dtype_kernel)[1](dtype_str="float8_e4m3fn")
    assert np.asarray(result).ravel()[0] == _FP8_E4M3FN_SENTINEL


@pytest.mark.fast
def test_resolve_fp8_e4m3_dtype_auto():
    """resolve_fp8_e4m3_dtype(AUTO) resolves to float8_e4m3fn under the gen4
    simulate trace (its nisa.get_nc_version() == gen4 branch)."""
    result = nki.simulate(_resolve_fp8_auto_kernel)[1]()
    assert np.asarray(result).ravel()[0] == _FP8_E4M3FN_SENTINEL


@nki.jit
def _apply_activation_kernel(x, act_val: int):
    """Run apply_activation for the given ActFnType value."""
    act_fn = ActFnType(act_val)
    xs = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    dst = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    kh.apply_activation(dst, xs, act_fn, scale=1.0)
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def _apply_activation_dx_kernel(x, act_val: int):
    """Run apply_activation_dx for the given ActFnType value."""
    act_fn = ActFnType(act_val)
    xs = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    dst = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    kh.apply_activation_dx(dst, xs, act_fn)
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=dst)
    return out


@pytest.mark.fast
def test_apply_activation_squared_relu():
    """apply_activation SquaredReLU path: relu then square (two instructions)."""
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float32)
    result = np.asarray(nki.simulate(_apply_activation_kernel)[1](x=x, act_val=ActFnType.SquaredReLU.value))
    expected = np.maximum(x, 0.0) ** 2
    assert np.allclose(result, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.fast
def test_apply_activation_single_instruction():
    """apply_activation else path: single-instruction activation (SiLU)."""
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float32)
    result = np.asarray(nki.simulate(_apply_activation_kernel)[1](x=x, act_val=ActFnType.SiLU.value))
    expected = x * (1.0 / (1.0 + np.exp(-x)))
    assert np.allclose(result, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.fast
@pytest.mark.parametrize(
    "act_fn",
    [ActFnType.SquaredReLU, ActFnType.SiLU, ActFnType.Swish],
)
def test_apply_activation_dx_runs(act_fn):
    """apply_activation_dx: each supported derivative op traces and produces finite output."""
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float32)
    result = np.asarray(nki.simulate(_apply_activation_dx_kernel)[1](x=x, act_val=act_fn.value))
    assert result.shape == (4, 8)
    assert np.all(np.isfinite(result))


@pytest.mark.fast
@pytest.mark.parametrize("use_manager", [False, True])
@pytest.mark.parametrize("buffer", [nl.sbuf, nl.shared_hbm])
def test_sbm_alloc_arguments(monkeypatch, use_manager, buffer):
    """Check allocator selection, argument forwarding, and returned tensor identity."""
    ndarray = Mock()
    monkeypatch.setattr(nl, "ndarray", ndarray)
    sbm = Mock(spec=SbufManager) if use_manager else None
    kwargs = {"shape": (4, 8), "dtype": nl.float16, "buffer": buffer, "name": "helper_tile"}

    result = kh._sbm_alloc(sbm, **kwargs, align=64)

    if sbm is not None:
        sbm.alloc_stack.assert_called_once_with(**kwargs, align=64)
        ndarray.assert_not_called()
        assert result is sbm.alloc_stack.return_value
    else:
        ndarray.assert_called_once_with(**kwargs)
        assert result is ndarray.return_value


@pytest.mark.fast
@pytest.mark.parametrize("use_manager", [False, True])
@pytest.mark.parametrize("address_offset", [0, 2048])
def test_psum_alloc_arguments(monkeypatch, use_manager, address_offset):
    """Only the manager path requests a manual PSUM address, including its offset."""
    ndarray = Mock()
    monkeypatch.setattr(nl, "ndarray", ndarray)
    sbm = Mock(spec=SbufManager) if use_manager else None

    result = kh._psum_alloc((4, 8), nl.float16, sbm, address_offset=address_offset)

    if use_manager:
        ndarray.assert_called_once_with((4, 8), dtype=nl.float16, buffer=nl.psum, address=(0, address_offset))
    else:
        ndarray.assert_called_once_with((4, 8), dtype=nl.float16, buffer=nl.psum)
    assert result is ndarray.return_value


@nki.jit
def _sbm_alloc_kernel(x, use_manager: bool):
    """Round-trip input through the helper's allocation before closing its scope."""
    sbm = None
    if use_manager:
        sbm = SbufManager(0, 200 * 1024, use_auto_alloc=True)
        sbm.open_scope(name="cov")
    tile = kh._sbm_alloc(sbm, x.shape, x.dtype, name="cov_tile", align=64)
    kernel_assert(tile.shape == x.shape, "Incorrect SBUF allocation shape")
    kernel_assert(tile.dtype == x.dtype, "Incorrect SBUF allocation dtype")
    kernel_assert(tile.buffer == nl.sbuf, "Incorrect SBUF allocation buffer")
    nisa.dma_copy(dst=tile, src=x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=tile)
    if sbm is not None:
        sbm.close_scope()
    return out


@nki.jit
def _psum_alloc_kernel(x, use_manager: bool):
    """Round-trip input through the helper's PSUM allocation and back to SBUF."""
    sbm = None
    if use_manager:
        sbm = SbufManager(0, 200 * 1024, use_auto_alloc=True)
    bank = kh._psum_alloc(x.shape, x.dtype, sbm, address_offset=2048)
    kernel_assert(bank.shape == x.shape, "Incorrect PSUM allocation shape")
    kernel_assert(bank.dtype == x.dtype, "Incorrect PSUM allocation dtype")
    kernel_assert(bank.buffer == nl.psum, "Incorrect PSUM allocation buffer")
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=x)
    nisa.tensor_copy(dst=bank, src=src)
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dst, src=bank)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=dst)
    return out


@pytest.mark.fast
@pytest.mark.parametrize("use_manager", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_sbm_alloc(use_manager, dtype):
    """Both allocation paths preserve every element of a nonconstant input."""
    x = (np.arange(32).reshape(4, 8) - 16).astype(dtype)
    result = np.asarray(nki.simulate(_sbm_alloc_kernel)[1](x=x, use_manager=use_manager))
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, x)


@pytest.mark.fast
@pytest.mark.parametrize("use_manager", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_psum_alloc(use_manager, dtype):
    """Manual-address and automatic PSUM allocations preserve the input data."""
    x = (np.arange(32).reshape(4, 8) - 16).astype(dtype)
    result = np.asarray(nki.simulate(_psum_alloc_kernel)[1](x=x, use_manager=use_manager))
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, x)
