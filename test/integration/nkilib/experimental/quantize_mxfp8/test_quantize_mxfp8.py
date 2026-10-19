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

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._dense_scales import get_dense_scale_shape
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    MX_PARTITION_SIZE,
    get_fp8_dtype,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import (
    quantize_block_mxfp8_kernel,
)

from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    QUANT_SCHEME_1X32,
    QUANT_SCHEME_WRAPX,
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

# Multi-tile K (several packing-group-sized spans) and non-divisible F; K=F=1024 runs LNC2.
MULTI_TILE_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 2048, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 3072, 2048, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 520, "float8_e4m3fn"),
    pytest.param("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn", marks=pytest.mark.fast),
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
    pytest.param("bfloat16", -1, 1, 128, 512, "float8_e4m3fn", marks=pytest.mark.fast),
    ("bfloat16", -1, 1, 256, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 384, 512, "float8_e4m3fn"),
    pytest.param("bfloat16", -1, 1, 640, 512, "float8_e4m3fn", marks=pytest.mark.fast),
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

# K aligned to MX_PARTITION_SIZE (32) but NOT to 128: exercises the sub-128 remainder tile,
# which is quantized padded up to a multiple of 32 partitions. Comments give the wrapX K tile
# decomposition; 1x32 (and K-by-F) loads the whole K % 512 remainder as one tile instead.
K_DIV_32_PARAMS = [
    ("bfloat16", -1, 1, 32, 512, "float8_e4m3fn"),  # 32
    ("bfloat16", -1, 1, 64, 512, "float8_e4m3fn"),  # 64
    ("bfloat16", -1, 1, 96, 512, "float8_e4m3fn"),  # 96
    pytest.param("bfloat16", -1, 1, 160, 512, "float8_e4m3fn", marks=pytest.mark.fast),  # 128 + 32
    ("bfloat16", -1, 1, 288, 512, "float8_e4m3fn"),  # 256 + 32
    ("bfloat16", -5, 5, 352, 512, "float8_e5m2"),  # 256 + 96
    ("bfloat16", -1, 1, 416, 1024, "float8_e4m3fn"),  # 256 + 128 + 32
    ("bfloat16", -1, 1, 480, 520, "float8_e4m3fn"),  # 256 + 128 + 96
    ("bfloat16", -200, 200, 544, 512, "float8_e4m3fn"),  # 512 + 32
    ("bfloat16", -1, 1, 1568, 512, "float8_e4m3fn"),  # 3x512 + 32
    ("bfloat16", -1, 1, 2016, 2344, "float8_e4m3fn"),  # 3x512 + 256 + 128 + 96, non-div F
    ("bfloat16", -1, 1, 2592, 1104, "float8_e4m3fn"),  # 5x512 + 32, non-div F
]

# Shapes for the 1x32 layout that stay on the 128-aligned K paths (the 1x32 analogue of the
# existing wrapX quick / non-divisible-F / non-divisible-K coverage).
QUANT_SCHEME_1X32_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 512, 1024, "float8_e4m3fn"),
    pytest.param("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn", marks=pytest.mark.fast),
    ("bfloat16", -5, 5, 2048, 2048, "float8_e5m2"),
    ("bfloat16", -200, 200, 512, 512, "float8_e4m3fn"),
    # F not a multiple of L_TILE_F: the short final F tile loads into a SUB_TILE-padded buffer
    ("bfloat16", -1, 1, 2048, 1152, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 1104, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 512, 128, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 2048, 2344, "float8_e4m3fn"),
    # K not a multiple of L_TILE_K, still 128-aligned
    ("bfloat16", -1, 1, 256, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 384, 512, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 640, 512, "float8_e4m3fn"),
    pytest.param("bfloat16", -5, 5, 768, 512, "float8_e5m2", marks=pytest.mark.fast),
    ("bfloat16", -1, 1, 1280, 1024, "float8_e4m3fn"),
]

# 1x32 LNC2 splits along F, so every shape needs F >= MIN_F_FOR_LNC2.
QUANT_SCHEME_1X32_LNC2_PARAMS = [
    ("bfloat16", -1, 1, 512, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1024, 2048, "float8_e4m3fn"),
    ("bfloat16", -5, 5, 2048, 1536, "float8_e5m2"),
    ("bfloat16", -1, 1, 640, 1024, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 480, 2048, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 1568, 1104, "float8_e4m3fn"),
]


# use_fast_dma_transpose: the fast (direct access pattern) DGT instead of the legacy
# vector-offset gather. The multi-tile-remainder shapes at large F (1696/2016 at F >= 4096) are the
# ones that exposed the uniform-destination-height requirement -- keep them.
FAST_DMA_PARAMS = [
    ("bfloat16", -1, 1, 512, 512, "float8_e4m3fn"),  # no remainder
    ("bfloat16", -1, 1, 1024, 1024, "float8_e4m3fn"),  # no remainder, 2 tiles
    pytest.param("bfloat16", -1, 1, 544, 512, "float8_e4m3fn", marks=pytest.mark.fast),  # 512 + 32
    ("bfloat16", -1, 1, 640, 4096, "float8_e4m3fn"),  # 512 + 128, many F tiles
    ("bfloat16", -5, 5, 96, 512, "float8_e5m2"),  # 96 alone -- one sub-128 tile
    ("bfloat16", -1, 1, 2048, 1104, "float8_e4m3fn"),  # no remainder, non-div F
    ("bfloat16", -1, 1, 1696, 4096, "float8_e4m3fn"),  # 3x512 + [128, 32] -- multi-tile remainder
    ("bfloat16", -1, 1, 1696, 5240, "float8_e4m3fn"),  # same, different F tiling
    ("bfloat16", -1, 1, 2016, 5240, "float8_e4m3fn"),  # 3x512 + [256, 128, 96]
    ("bfloat16", -1, 1, 480, 520, "float8_e4m3fn"),  # [256, 128, 96] with no full tile
]

# Fast DMA transpose with LNC2: F is halved at a SUB_TILE boundary, so a core can start or end
# mid-tile, and a core with F <= 512 uses half-width (256) F tiles.
FAST_DMA_LNC2_PARAMS = [
    ("bfloat16", -1, 1, 2048, 1104, "float8_e4m3fn"),  # cores [0, 640) = 512 + 128, [640, 1104) = 256 + 208
    ("bfloat16", -1, 1, 1696, 1024, "float8_e4m3fn"),  # 512 per core -> 2 x 256; 3x512 + [128, 32]
    ("bfloat16", -1, 1, 480, 1040, "float8_e4m3fn"),  # [256, 128, 96]; cores 640 and 400
    ("bfloat16", -5, 5, 2016, 5240, "float8_e5m2"),  # 3x512 + [256, 128, 96]; cores 2688 and 2552
]

# K x F large enough that a core's scales exceed _SCALE_SBUF_BUDGET_BYTES, so the kernel stores
# them in several F-tile groups (e.g. Qwen3-32B gate/up [25600, 5120] at LNC1: 6 groups).
LARGE_KF_PARAMS = [
    ("bfloat16", -1, 1, 5120, 25600, "float8_e4m3fn"),
    ("bfloat16", -1, 1, 8192, 8192, "float8_e4m3fn"),
]


# (F, shard0 K, shard1 K): two K-shards quantized separately and concatenated on dim 0 must match
# quantizing the whole [F, K0 + K1] tensor. Shard K is a multiple of 32 but not of 128 / 512, the
# case the old tile-structured scale layouts could not concatenate (e.g. Qwen3-32B's per-rank
# out_features 160 / 416 at 64-way FSDP).
K_SHARD_CONCAT_PARAMS = [
    pytest.param(512, 160, 160, marks=pytest.mark.fast),
    (512, 416, 416),
    (520, 608, 1440),  # uneven shards, non-L_TILE_F-aligned F
    (1024, 1696, 1696),  # 3x512 + [128, 32] per shard
    (512, 96, 32),
    (1024, 2016, 544),
    (512, 2048, 2048),  # control: already aligned
]


@nki.jit
@with_active_sbm
def quantize_k_shards_and_concat_kernel(
    shard0: nl.NkiTensor,
    shard1: nl.NkiTensor,
    return_fp8_dtype: str,
    quant_scheme: str,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Quantize two [F, K_i] K-shards separately, then concatenate data and scales along dim 0."""
    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope(name="TEST K-SHARD CONCAT")

    F, K0 = shard0.shape
    K1 = shard1.shape[1]
    quantized_data_hbm = nl.ndarray(
        ((K0 + K1) // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR),
        dtype=get_fp8_dtype(return_fp8_dtype),
        buffer=nl.shared_hbm,
    )
    quantized_scales_hbm = nl.ndarray(get_dense_scale_shape(K0 + K1, F), dtype=nl.float8_e8m0fnu, buffer=nl.shared_hbm)
    data_row = 0
    scale_row = 0
    for shard in (shard0, shard1):
        K_shard = shard.shape[1]
        shard_scales, shard_data = quantize_block_mxfp8_kernel(shard, return_fp8_dtype, False, quant_scheme)
        nisa.dma_copy(dst=quantized_data_hbm[nl.ds(data_row, K_shard // INTERLEAVE_FACTOR), :], src=shard_data)
        nisa.dma_copy(dst=quantized_scales_hbm[nl.ds(scale_row, K_shard // MX_PARTITION_SIZE), :], src=shard_scales)
        data_row += K_shard // INTERLEAVE_FACTOR
        scale_row += K_shard // MX_PARTITION_SIZE

    sbm.close_scope()
    return quantized_scales_hbm, quantized_data_hbm


def quantize_k_shards_and_concat_torch_ref(
    shard0: np.ndarray, shard1: np.ndarray, return_fp8_dtype: str, quant_scheme: str
) -> dict[str, np.ndarray]:
    """Golden for quantize_k_shards_and_concat_kernel: quantize the WHOLE [F, K0 + K1] tensor at once."""
    return quantize_block_mxfp8_torch_ref(
        np.concatenate([shard0, shard1], axis=1), return_fp8_dtype, quant_scheme=quant_scheme
    )


def filter_illegal_combinations(
    input_dtype: str,
    K: int,
    F: int,
    return_fp8_dtype: str,
    lnc_degree: int,
    quant_scheme: str,
) -> FilterResult:
    if K % 32 != 0 or F % 8 != 0:
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
    platform_target: Platforms,
    is_negative_test: bool = False,
    quant_scheme: str = QUANT_SCHEME_WRAPX,
    use_fast_dma_transpose: bool = False,
    src_orientation: TensorOrientation | None = None,
):
    if not platform_target.is_trn3():
        pytest.skip("MX is only supported on TRN3.")

    run_with_lnc2 = lnc_degree > 1

    captured_input = generate_quantize_mxfp8_inputs(
        input_dtype,
        input_range_low,
        input_range_high,
        K,
        F,
        return_fp8_dtype,
        run_with_lnc2,
        quant_scheme,
        use_fast_dma_transpose,
        src_orientation,
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
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, MULTI_TILE_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_multi_tile(
        self,
        test_manager,
        platform_target,
        input_dtype,
        input_range_low,
        input_range_high,
        K,
        F,
        return_fp8_dtype,
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
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, K_DIV_32_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_k_div_32(
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
            platform_target,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, QUANT_SCHEME_1X32_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, K_DIV_32_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_k_div_32(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, QUANT_SCHEME_1X32_LNC2_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_lnc2(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, FAST_DMA_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_fast_dma_transpose(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        """wrapX with use_fast_dma_transpose=True -- the fast DGT instead of legacy / PE."""
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            platform_target,
            use_fast_dma_transpose=True,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, FAST_DMA_LNC2_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_fast_dma_transpose_lnc2(
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
            platform_target,
            use_fast_dma_transpose=True,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, LARGE_KF_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_large(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, LARGE_KF_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_k_by_f_large(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
            src_orientation=TensorOrientation.K_BY_F,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, QUANT_SCHEME_1X32_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_k_by_f(
        self, test_manager, platform_target, input_dtype, input_range_low, input_range_high, K, F, return_fp8_dtype
    ) -> None:
        """1x32 with a [K, F] input (src_orientation=K_BY_F): same output as the [F, K] transpose."""
        _run_test(
            test_manager,
            input_dtype,
            input_range_low,
            input_range_high,
            K,
            F,
            return_fp8_dtype,
            1,
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
            src_orientation=TensorOrientation.K_BY_F,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, K_DIV_32_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_k_by_f_k_div_32(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
            src_orientation=TensorOrientation.K_BY_F,
        )

    @pytest_parametrize(_BASE_PARAM_NAMES, QUANT_SCHEME_1X32_LNC2_PARAMS, abbrevs=_BASE_ABBREVS)
    def test_quantize_mxfp8_1x32_k_by_f_lnc2(
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
            platform_target,
            quant_scheme=QUANT_SCHEME_1X32,
            src_orientation=TensorOrientation.K_BY_F,
        )

    @pytest.mark.fast
    def test_quantize_mxfp8_wrapx_k_by_f_rejected(self, test_manager, platform_target) -> None:
        """wrapX loads F-by-K only, so a K_BY_F input must be rejected rather than misread."""
        _run_test(
            test_manager,
            "bfloat16",
            -1,
            1,
            1024,
            512,
            "float8_e4m3fn",
            1,
            platform_target,
            is_negative_test=True,
            quant_scheme=QUANT_SCHEME_WRAPX,
            src_orientation=TensorOrientation.K_BY_F,
        )

    @pytest.mark.parametrize("F, K0, K1", K_SHARD_CONCAT_PARAMS)
    def test_quantize_mxfp8_k_shard_concat(self, test_manager, platform_target, F, K0, K1) -> None:
        """1x32: quantize(shard0) ++ quantize(shard1) along dim 0 == quantize(whole), bit for bit."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        full_input = generate_quantize_mxfp8_inputs(
            "bfloat16", -100, 100, K0 + K1, F, "float8_e4m3fn", False, QUANT_SCHEME_1X32
        )
        full = full_input["src_tensor"]
        kernel_input = {
            "shard0": np.ascontiguousarray(full[:, :K0]),
            "shard1": np.ascontiguousarray(full[:, K0:]),
            "return_fp8_dtype": "float8_e4m3fn",
            "quant_scheme": QUANT_SCHEME_1X32,
        }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=quantize_k_shards_and_concat_kernel,
            torch_ref=quantize_k_shards_and_concat_torch_ref,
            kernel_input_generator=lambda _config: kernel_input,
            output_tensor_descriptor=lambda _input: build_output_tensors(full_input),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            # Golden is the WHOLE tensor quantized at once.
            custom_validation_args=build_custom_validation_args(full_input),
        )

    @pytest.mark.coverage_parametrize(
        K=np.random.choice(np.arange(512, 8193, 32), 10, replace=False),
        F=np.random.choice(np.arange(512, 8193, 8), 10, replace=False),
        input_dtype=["bfloat16"],
        return_fp8_dtype=["float8_e4m3fn", "float8_e5m2"],
        lnc_degree=BoundedRange([1, 2], boundary_values=[]),
        quant_scheme=[QUANT_SCHEME_WRAPX, QUANT_SCHEME_1X32],
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
        quant_scheme,
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
            platform_target,
            is_negative_test=is_negative_test_case,
            quant_scheme=quant_scheme,
        )
