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

"""Test suite for quantize_mxfp8 kernel using UnitTestFramework with custom validators."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import (
    quantize_block_mxfp8_kernel,
)

from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    build_custom_validation_args,
    build_output_tensors,
    generate_quantize_mxfp8_inputs,
    quantize_block_mxfp8_torch_ref,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

# ============================================================================
# Test Parameters
# ============================================================================

_BASE_PARAM_NAMES = "input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype"
_BASE_ABBREVS = {"input_dtype": "dt", "input_range_low": "lo", "input_range_high": "hi", "return_fp8_dtype": "fp8"}

SINGLE_TEST_PARAMS = [("bfloat16", -100, 100, 4096, 2048, "float8_e4m3fn")]

QUICK_TEST_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 512, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 1152, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 1104, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 128, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 2344, "float8_e4m3fn"),
]

SCALE_PACKING_PARAM_NAMES = _BASE_PARAM_NAMES + ", enable_scale_packing"
SCALE_PACKING_ABBREVS = {**_BASE_ABBREVS, "enable_scale_packing": "pack"}
SCALE_PACKING_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn", True),
    ("bfloat16", -1, 1, 2048, 2048, "float8_e4m3fn", True),
    ("bfloat16", -1, 1, 3072, 2048, "float8_e4m3fn", True),
    ("bfloat16", -1, 1, 1024, 520, "float8_e4m3fn", True),
    pytest.param("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn", True, marks=pytest.mark.fast),
]

INPUT_RANGE_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -200, 200, 512, 512, "float8_e4m3fn"),
]

FP8_DTYPE_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 512, 512, "float8_e5m2"),
    ("bfloat16", -5, 5, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 512, 512, "float8_e5m2"),
]

NON_DIVISIBLE_F_PARAMS = [
    ("bfloat16", -1, 1, 512, 640, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 512, 768, "float8_e5m2"),
    ("bfloat16", -1, 1, 1024, 1152, "float8_e4m3fn"),
    ("bfloat16", -200, 200, 2048, 1104, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 2344, "float8_e4m3fn"),
]

LNC2_SPLIT_F_PARAMS = [
    ("bfloat16", -1, 1, 512, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 2048, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 2048, 2048, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 512, 1536, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 1024, 1536, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 2560, "float8_e5m2"),
    ("bfloat16", -5, 5, 2048, 3584, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 1024, "float8_e5m2"),
    ("bfloat16", -5, 5, 1024, 2048, "float8_e5m2"),
]

NON_DIVISIBLE_K_PARAMS = [
    ("bfloat16", -1, 1, 128, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 256, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 384, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 640, 512, "float8_e4m3fn"),
    pytest.param("bfloat16", -5, 5, 768, 512, "float8_e5m2", marks=pytest.mark.fast),
    ("bfloat16", -1, 1, 896, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1152, 512, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 1280, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2176, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 4864, 520, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 4736, 520, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 768, 2048, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 4864, 6144, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 3328, 6144, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1920, 3000, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 4992, 1288, "float8_e4m3fn"),
]

# Todo add back in K%512=384 tests when bug is fixed
NON_DIVISIBLE_K_SCALE_PACKING_PARAMS = [
    pytest.param("bfloat16", -1, 1, 128, 512, "float8_e4m3fn", marks=pytest.mark.fast),
    ("bfloat16", -1, 1, 256, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 384, 512, "float8_e4m3fn"),
    pytest.param("bfloat16", -1, 1, 640, 512, "float8_e4m3fn", marks=pytest.mark.fast),
    pytest.param("bfloat16", -5, 5, 768, 512, "float8_e5m2", marks=pytest.mark.fast),
    ("bfloat16", -1, 1, 896, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1152, 512, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 1280, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2176, 512, "float8_e4m3fn"),
]


def filter_illegal_combinations(
    input_dtype: str, K: int, F: int, return_fp8_dtype: str, lnc_degree: int, enable_scale_packing: bool
) -> FilterResult:
    if K % 128 != 0 or F % 8 != 0:
        return FilterResult.INVALID
    return FilterResult.VALID


# ============================================================================
# Helper to run a single test via UnitTestFramework
# ============================================================================


def _run_test(
    test_manager: Orchestrator,
    input_dtype: str,
    input_range_low: float,
    input_range_high: float,
    K: int,
    F: int,
    return_fp8_dtype: str,
    lnc_degree: int,
    enable_scale_packing: bool,
    platform_target: Platforms,
    is_negative_test: bool = False,
):
    if not platform_target.is_trn3():
        pytest.skip("MX is only supported on TRN3.")

    run_with_lnc2 = lnc_degree > 1

    captured_input = generate_quantize_mxfp8_inputs(
        input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype, run_with_lnc2, enable_scale_packing
    )

    def input_generator(test_config):
        return captured_input

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=quantize_block_mxfp8_kernel,
        torch_ref=quantize_block_mxfp8_torch_ref,
        kernel_input_generator=input_generator,
        output_tensor_descriptor=build_output_tensors,
    )

    compiler_args = CompilerArgs(
        logical_nc_config=lnc_degree,
        platform_target=platform_target,
    )

    custom_val = None if is_negative_test else build_custom_validation_args(captured_input)

    framework.run_test(
        test_config=None,
        compiler_args=compiler_args,
        is_negative_test=is_negative_test,
        custom_validation_args=custom_val,
    )


# ============================================================================
# Test Class
# ============================================================================


@pytest_test_metadata(name="Quantize MXFP8")
@pytest_marks(["quantize_mxfp8", "mx", "mxfp8"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestQuantizeMxfp8Kernel:
    """Test suite for quantize_mxfp8 kernel with comprehensive parameter coverage."""

    @pytest_parametrize(_BASE_PARAM_NAMES, SINGLE_TEST_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_single(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            2,
            True,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, QUICK_TEST_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_quick(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            False,
            platform_target,
        )

    @pytest_parametrize(SCALE_PACKING_PARAM_NAMES, SCALE_PACKING_PARAMS, abbrevs=SCALE_PACKING_ABBREVS)
    def test_quantize_mxfp8_scale_packing(
        self,
        test_manager,
        platform_target,
        input_dtype,
        input_range_low,
        input_range_high,
        K,
        F,
        return_fp8_dtype,
        enable_scale_packing,
    ) -> None:
        lnc_degree = 2 if (K == 1024 and F == 1024) else 1
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            lnc_degree,
            enable_scale_packing,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, INPUT_RANGE_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_input_ranges(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            False,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, FP8_DTYPE_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_fp8_dtypes(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            False,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, NON_DIVISIBLE_F_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_non_divisible_f(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            False,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, NON_DIVISIBLE_K_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_non_divisible_k(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            False,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, NON_DIVISIBLE_K_SCALE_PACKING_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_non_divisible_k_scale_packing(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            True,
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, LNC2_SPLIT_F_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_lnc2(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            2,
            True,
            platform_target,
        )

    # TODO: update K divisibility for 128 when compiler bug is fixed
    @pytest.mark.coverage_parametrize(
        K=np.random.choice(np.arange(512, 8193, 128), 10, replace=False),
        F=np.random.choice(np.arange(512, 8193, 8), 10, replace=False),
        input_dtype=["bfloat16"],
        return_fp8_dtype=["float8_e4m3fn", "float8_e5m2"],
        lnc_degree=BoundedRange([1, 2], boundary_values=[]),
        enable_scale_packing=[True, False],
        filter=filter_illegal_combinations,
        coverage="pairs",
    )
    def test_quantize_mxfp8_sweep(
        self,
        test_manager,
        platform_target,
        input_dtype,
        K,
        F,
        return_fp8_dtype,
        lnc_degree,
        enable_scale_packing,
        is_negative_test_case,
    ):
        INPUT_RANGE_MIN = -1000
        INPUT_RANGE_MAX = 1000
        _run_test(
            test_manager,
            input_dtype,
            INPUT_RANGE_MIN,
            INPUT_RANGE_MAX,
            K,
            F,
            return_fp8_dtype,
            lnc_degree,
            enable_scale_packing,
            platform_target,
            is_negative_test=is_negative_test_case,
        )
