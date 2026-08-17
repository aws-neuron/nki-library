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

"""Integration tests for cumsum kernel.

All correctness checks validate the kernel's hardware output against the torch
reference (`torch.cumsum`, computed in fp32 then cast to the input dtype), which
mirrors what the kernel does internally (fp32 accumulate, cast to output dtype).

The kernel selects its scan instruction by dtype: tensor_scalar_cumulative for
its ISA-allowed dtypes, tensor_tensor_scan fallback for int32/uint32 (which the
former rejects at compile time). `test_cumsum_dtype_sweep` exercises every dtype
the kernel advertises — floats, fp8, and the signed/unsigned ints — so a
dtype-domain regression (e.g. the CR-290142114 int32 compile failure) is caught.

Inputs are bounded per dtype so the reference stays an exact or tightly-bounded
oracle on real hardware rather than a brittle one:
  - integers: values bounded so the running sum cannot overflow, keeping torch
    (which promotes ints to int64) exact.
  - fp8: values bounded so every partial sum is an integer that fp8 represents
    exactly (see `_exact_int_bound`), so the comparison is exact with no
    rounding ambiguity at all.
  - float32/float16/bfloat16: tolerance scaled to mantissa width and scan
    length, since fp32 accumulation rounding is genuine and unavoidable.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.core.cumsum import cumsum
from nkilib_src.nkilib.core.cumsum.cumsum_torch import cumsum_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Every storable input dtype cumsum accepts. This is the full set nl exports minus
# dtypes that are not meaningful cumsum *correctness* inputs:
#   - tfloat32: a compute mode, not a storage dtype (a tf32 input is stored as
#     float32, already covered here).
#   - bool_: a running sum of booleans is not a meaningful cumsum input.
# All dtypes below take the fast tensor_scalar_cumulative path EXCEPT int32/uint32,
# which fall back to tensor_tensor_scan (tensor_scalar_cumulative rejects them at
# compile time).
_FP8_DTYPES = (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2)

_ALL_DTYPES = [
    np.float32,
    np.float16,
    ml_dtypes.bfloat16,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e5m2,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]


def _exact_int_bound(dtype):
    """Largest N such that every integer in [0, N] is exactly representable in `dtype`.

    A binary float with an m-bit mantissa field represents consecutive integers
    exactly up to 2**(m+1) (the implicit leading bit gives one extra bit of
    precision); at 2**(m+1)+1 the spacing becomes 2 and odd integers are lost.
    For fp8 that is 16 (e4m3, m=3) and 8 (e5m2, m=2).

    This is what makes an fp8 cumsum check exact rather than tolerance-based: if
    every partial sum is an integer <= this bound, the fp8 cast is lossless, so
    hardware and the torch reference must agree bit-for-bit regardless of
    accumulation order.

    `ml_dtypes.finfo` is used rather than `np.finfo`, which rejects fp8 types.
    """
    return 2 ** (ml_dtypes.finfo(dtype).nmant + 1)


def filter_invalid_combinations(batch, hidden, ndim, seq_len, dtype=None):
    """Filter out invalid parameter combinations.

    For 2D inputs, seq_len doesn't matter - only run one combination.
    """
    if ndim == 2 and seq_len != 1:
        return FilterResult.REDUNDANT
    return FilterResult.VALID


def _generate_inputs(batch, hidden, ndim, seq_len, dtype):
    """Generate cumsum kernel inputs from parameters (float dtypes)."""
    np.random.seed(42)
    shape = (batch, seq_len, hidden) if ndim == 3 else (batch, hidden)
    return {"x": np.random.randn(*shape).astype(dtype)}


def _sparse_pm1(shape, n, nnz, lo, dtype):
    """Draw values in [lo, 1] but zero out all but the first `nnz` positions along
    the scanned (last) axis, so the running sum along that axis never exceeds nnz.

    Used for both the integer and fp8 sweeps. For long axes on narrow dtypes this
    makes the axis sparse; the specific values don't matter for a correctness
    check, only that the running sum stays inside the exactly-representable range.
    """
    vals = np.random.randint(lo, 2, size=shape).astype(dtype)
    if nnz < n:
        # Zero via in-place slice assignment rather than np.where(mask, vals, 0):
        # a Python int cannot be promoted against an fp8 array, which raises
        # DTypePromotionError for the fp8 dtypes.
        vals[..., nnz:] = dtype(0)
    return vals


def _generate_inputs_any_dtype(batch, hidden, ndim, seq_len, dtype):
    """Generate inputs for any ML dtype, bounded so the running cumsum stays
    exactly representable in that dtype. This keeps torch.cumsum a valid oracle
    on real hardware: integers and fp8 are compared exactly, and the wider floats
    only carry genuine fp32 accumulation rounding.
    """
    np.random.seed(42)
    shape = (batch, seq_len, hidden) if ndim == 3 else (batch, hidden)
    n = seq_len * hidden if ndim == 3 else hidden  # length of the scanned axis
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        # Keep the running sum within the dtype range (no overflow) so torch is
        # an exact oracle.
        nnz = max(1, min(n, info.max // 2))
        return {"x": _sparse_pm1(shape, n, nnz, 0 if info.min == 0 else -1, dtype)}
    if dtype in _FP8_DTYPES:
        # Bound the running sum to fp8's exactly-representable integer range (16
        # for e4m3, 8 for e5m2). Every partial sum is then integral and lossless
        # in fp8, so hardware must match torch bit-for-bit no matter what order
        # the accumulation happened in -- turning fp8 from the least checkable
        # dtype into an exactly checkable one.
        return {"x": _sparse_pm1(shape, n, max(1, min(n, _exact_int_bound(dtype))), -1, dtype)}
    return {"x": np.random.randn(*shape).astype(dtype)}


def _tolerances(dtype, hidden):
    """Per-dtype (rtol, atol) for the torch comparison.

    Integers and fp8 are exact: their inputs are bounded (see
    `_generate_inputs_any_dtype`) so every partial sum is exactly representable,
    leaving no room for rounding to differ between hardware and the reference.
    The wider floats keep a tolerance because their inputs are unbounded normals,
    so fp32 accumulation rounding is real; it grows with the scanned length and
    shrinks with mantissa width.
    """
    if np.issubdtype(dtype, np.integer) or dtype in _FP8_DTYPES:
        return 0.0, 0.0
    if dtype == ml_dtypes.bfloat16:
        return 1e-2, 1e-2
    # float32 / float16: hidden > 5000 accumulates enough fp32 error to need 1e-2
    return 1e-3, (1e-2 if hidden > 5000 else 1e-3)


def _output_tensors(kernel_input):
    """Generate output tensor descriptors."""
    return {"output_0": np.zeros_like(kernel_input["x"])}


# fmt: off
FAST_PARAM_NAMES = \
    "batch, hidden, ndim, seq_len, dtype"
FAST_TEST_PARAMS = [
    pytest.param(1,   256,  2, 1, np.float32,         id="1_256_2_1_float32",         marks=pytest.mark.fast),
    pytest.param(64,  1024, 3, 4, ml_dtypes.bfloat16, id="64_1024_3_4_bfloat16"),
    pytest.param(128, 2048, 2, 1, np.float32,         id="128_2048_2_1_float32"),
    # int32 and uint32 both take the tensor_tensor_scan fallback (TSCR rejects
    # them). These vectors guard that the fallback keeps compiling for both — a
    # compile-time assertion regression (the CR-290142114 escape) fails here
    # even in compile-only mode.
    pytest.param(1,   15,   2, 1, np.int32,           id="1_15_2_1_int32",            marks=pytest.mark.fast),
    pytest.param(1,   15,   2, 1, np.uint32,          id="1_15_2_1_uint32",           marks=pytest.mark.fast),
    # fp8 shares the tensor_scalar_cumulative path with the wider floats, but keep
    # a fast vector per fp8 type so a dtype-domain regression that rejects fp8 at
    # compile time fails here too, not only in the (slower) dtype sweep.
    pytest.param(1,   15,   2, 1, ml_dtypes.float8_e4m3fn, id="1_15_2_1_float8_e4m3fn", marks=pytest.mark.fast),
    pytest.param(1,   15,   2, 1, ml_dtypes.float8_e5m2,   id="1_15_2_1_float8_e5m2",   marks=pytest.mark.fast),
]
# fmt: on


@pytest_test_metadata(name="Cumsum")
@pytest_marks(["cumsum"])
class TestCumsumKernel:
    """Test class for cumsum kernel."""

    @pytest.mark.parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_cumsum_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, hidden, ndim, seq_len, dtype
    ):
        """Fast compile-only tests with minimal coverage."""
        rtol, atol = _tolerances(dtype, hidden)

        def input_generator(test_config):
            return _generate_inputs_any_dtype(batch, hidden, ndim, seq_len, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cumsum,
            torch_ref=torch_ref_wrapper(cumsum_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128, 256, 512, 1024, 2048],
        hidden=[256, 512, 1024, 2048, 4096, 6144, 8192],
        ndim=[2, 3],
        seq_len=[1, 4, 8, 10],
        dtype=[np.float32, np.float16, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_cumsum_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        hidden,
        ndim,
        seq_len,
        dtype,
        is_negative_test_case,
    ):
        """Float sweep with pairwise coverage against the torch reference."""
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return _generate_inputs(batch, hidden, ndim, seq_len, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cumsum,
            torch_ref=torch_ref_wrapper(cumsum_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-2 if hidden > 5000 else 1e-3,
            rtol=1e-2 if is_bf16 else 1e-3,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128, 256, 512, 1024, 2048],
        hidden=[256, 512, 1024, 2048, 4096, 6144, 8192],
        ndim=[2, 3],
        seq_len=[1, 4, 8, 10],
        dtype=_ALL_DTYPES,
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_cumsum_dtype_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        hidden,
        ndim,
        seq_len,
        dtype,
        is_negative_test_case,
    ):
        """Every dtype the kernel advertises vs the torch reference, bounded inputs.

        Covers the full advertised dtype domain — floats, both fp8 types, and the
        signed/unsigned ints including the int32/uint32 tensor_tensor_scan
        fallback — so a dtype-domain regression (the CR-290142114 int32 compile
        escape) is caught. Inputs are bounded per dtype so torch is an exact
        oracle for ints and fp8 and a tolerance oracle for the wider floats; see
        `_generate_inputs_any_dtype` and `_tolerances`.
        """
        rtol, atol = _tolerances(dtype, hidden)

        def input_generator(test_config):
            return _generate_inputs_any_dtype(batch, hidden, ndim, seq_len, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cumsum,
            torch_ref=torch_ref_wrapper(cumsum_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=atol,
            rtol=rtol,
            is_negative_test=is_negative_test_case,
        )
