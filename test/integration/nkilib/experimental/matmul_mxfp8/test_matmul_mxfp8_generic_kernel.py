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

"""Integration tests for MXFP8 matrix multiplication kernel."""

import json
import os
import random
from typing import Any

import neuron_dtypes as dtype
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from neuronxcc.nki._private.private_api import float8_e4m3fn_x4, float8_e5m2_x4
from neuronxcc.nki._private.test import mx_util
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from typing_extensions import override

from test.integration.nkilib.experimental.matmul_mxfp8 import (
    config_helper,
    constants,
    model_config_reader,
    random_input_generator,
)
from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    Q_TILE_K,
    generate_golden_packed_scales,
)
from test.utils import common_dataclasses, coverage_parametrized_tests, test_orchestrator
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

"""Golden generation functions for MXFP8 matrix multiplication test validation."""

# Load GPU thresholds once at module level
GPU_THRESHOLD_FILE = os.path.join(os.path.dirname(__file__), "gpu_threshold.json")
GPU_THRESHOLDS = {}
if os.path.exists(GPU_THRESHOLD_FILE):
    with open(GPU_THRESHOLD_FILE, 'r') as file_handle:
        GPU_THRESHOLDS = json.load(file_handle)


def get_mx_max_exp(dst_dtype: Any) -> int:
    """Get maximum exponent value for MX format based on dtype.

    Args:
        dst_dtype: Target float8 dtype (float8_e5m2_x4 or float8_e4m3fn_x4).

    Returns:
        int: Maximum exponent value (14 for E5M2, 7 for E4M3FN).

    Raises:
        AssertionError: If dst_dtype is not supported.
    """
    # subtracted one for TRN3 rounding mode
    MAX_EXP_E5M2 = 15 - 1
    MAX_EXP_E4M3FN = 8 - 1
    max_exp_values = {float8_e5m2_x4: MAX_EXP_E5M2, float8_e4m3fn_x4: MAX_EXP_E4M3FN}
    assert dst_dtype in max_exp_values, f"no max exp value provided for {dst_dtype}"
    return max_exp_values.get(dst_dtype)


def get_dists(num: int = 10, edge_rate: float = 0.3, seed: int = None):
    """Get random distributions for test input generation.

    Args:
        num: Number of distributions to generate. Default is 10.
        edge_rate: Probability of generating edge case values. Default is 0.3.
        seed: Random seed for reproducibility. Default is None.

    Returns:
        list: List of (distribution_name, json_params) tuples.
    """
    dists = random_input_generator.get_random_distributions(num, None, edge_rate, seed)
    dists = [
        (dist, json.dumps({k: float(f"{v:.3g}") if isinstance(v, float) else v for k, v in param.items()}))
        for (dist, param) in dists
    ]
    return dists


"""Test parameter grids for MXFP8 matrix multiplication tests."""

TESTS_PER_SETUP = 3
SWEEP_NUM_VALUES = 50


def populate_tests(grid):
    """Expand each TestConfig into TESTS_PER_SETUP generated subconfigs.

    If a grid entry has `fast_subset` set, the listed sub-indices (relative
    to that grid entry's autoGenerateRandomSubset output, not flat indices)
    are wrapped with pytest.mark.fast. Per-entry inline marking is stable
    to grid reordering and additions; only seed and TESTS_PER_SETUP changes
    can shift the sub-index assignment.
    """
    res = []
    for conf in grid:
        sub = conf.autoGenerateRandomSubset(TESTS_PER_SETUP)
        fast = getattr(conf, "fast_subset", frozenset())
        for i, t in enumerate(sub):
            if i in fast:
                res.append(pytest.param(t, marks=pytest.mark.fast))
            else:
                res.append(t)
    return res


# Base grid
GRID = [
    config_helper.TestConfig(M=512, K=512, N=512, description="Edge cases - single block", seed=52, fast_subset={0, 1}),
    config_helper.TestConfig(
        M=1024, K=1024, N=512, description="Small matrices, various tile counts", seed=52, fast_subset={2}
    ),
    config_helper.TestConfig(
        M=2048, K=2048, N=2048, description="Medium matrices, different tile counts", seed=52, fast_subset={0}
    ),
    config_helper.TestConfig(
        M=1024, K=2048, N=512, description="Non-square matrices with different tile counts", seed=52
    ),
    config_helper.TestConfig(
        M=512, K=1024, N=1024, description="Non-square matrices with different tile counts", seed=52
    ),
    config_helper.TestConfig(
        M=4096, K=1024, N=2048, description="Non-square matrices with different tile counts", seed=52
    ),
]

# Large grid (excluded from fast suite due to compile time)
GRID_LARGE = [
    config_helper.TestConfig(M=4096, K=8192, N=4096, description="Large matrices", seed=52),
]


GRID_SWEEP_M = [
    config_helper.TestConfig(M=32, K=512, N=256, description="Small M dimension: M=32, K=512, N=256", seed=52),
    config_helper.TestConfig(M=96, K=512, N=4096, description="Small M dimension: M=96, K=512, N=4096", seed=52),
    config_helper.TestConfig(M=384, K=512, N=512, description="Medium M dimension: M=384, K=512, N=512", seed=52),
    config_helper.TestConfig(M=640, K=1024, N=1024, description="Medium M dimension: M=640, K=1024, N=1024", seed=52),
    config_helper.TestConfig(M=1536, K=2048, N=2048, description="Large M dimension: M=1536, K=2048, N=2048", seed=52),
    config_helper.TestConfig(M=3072, K=2048, N=2048, description="Large M dimension: M=3072, K=2048, N=2048", seed=52),
]

GRID_SWEEP_N = [
    config_helper.TestConfig(M=512, K=512, N=256, description="Small N dimension: M=512, K=512, N=256", seed=52),
    config_helper.TestConfig(M=512, K=512, N=768, description="Medium N dimension: M=512, K=512, N=768", seed=52),
    config_helper.TestConfig(M=1024, K=1024, N=1536, description="Medium N dimension: M=1024, K=1024, N=1536", seed=52),
    config_helper.TestConfig(M=2048, K=2048, N=3072, description="Large N dimension: M=2048, K=2048, N=3072", seed=52),
    config_helper.TestConfig(
        M=2048, K=2048, N=6144, description="Very large N dimension: M=2048, K=2048, N=6144", seed=52
    ),
]

GRID_SWEEP_K = [
    config_helper.TestConfig(M=512, K=256, N=512, description="Small K dimension: M=512, K=256, N=512", seed=52),
    config_helper.TestConfig(M=512, K=384, N=512, description="Small K dimension: M=512, K=384, N=512", seed=52),
    config_helper.TestConfig(M=512, K=768, N=512, description="Medium K dimension: M=512, K=768, N=512", seed=52),
    config_helper.TestConfig(M=512, K=1536, N=512, description="Medium K dimension: M=512, K=1536, N=512", seed=52),
    config_helper.TestConfig(M=1024, K=2560, N=1024, description="Large K dimension: M=1024, K=2560, N=1024", seed=52),
    config_helper.TestConfig(M=2048, K=3584, N=2048, description="Large K dimension: M=2048, K=3584, N=2048", seed=52),
]

GRID_ALL_NON_DIVISIBLE = [
    config_helper.TestConfig(M=96, K=1600, N=768, description="Small M, medium K and N: M=96, K=1600, N=768", seed=52),
    config_helper.TestConfig(
        M=640, K=1600, N=3008, description="Medium M, medium K, large N: M=640, K=1600, N=3008", seed=52
    ),
    config_helper.TestConfig(
        M=32, K=1600, N=1600, description="Small M, medium K and N: M=32, K=1600, N=1600", seed=52
    ),
    config_helper.TestConfig(M=128, K=256, N=256, description="Small M and N: M=128, K=256, N=256", seed=52),
]

GRID_PARTIAL_NON_DIVISIBLE = [
    config_helper.TestConfig(M=128, K=1024, N=256, description="Small M and N: M=128, K=1024, N=256", seed=52),
    config_helper.TestConfig(M=4096, K=256, N=256, description="Small M and N: M=4096, K=256, N=256", seed=52),
]

# Fast subset of non-divisible configs to ensure remainder/masking code path
# is covered by -m fast runs (KTK-118)
GRID_NON_DIVISIBLE_FAST = [
    config_helper.TestConfig(M=640, K=1600, N=3008, description="Fast non-divisible: all dims non-aligned", seed=52),
    config_helper.TestConfig(M=4096, K=256, N=256, description="Fast non-divisible: partial non-aligned", seed=52),
]

GRID_PREQUANTIZED = [
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        description="Prequantized: LHS=BF16, RHS=BF16",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        description="Prequantized: LHS=MXFP8_X4, RHS=BF16",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        description="Prequantized: LHS=BF16, RHS=MXFP8_X4",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        description="Prequantized: LHS=MXFP8, RHS=BF16",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.MXFP8,
        description="Prequantized: LHS=BF16, RHS=MXFP8",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        rhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        description="Prequantized: LHS=MXFP8_X4, RHS=MXFP8_X4",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.MXFP8,
        description="Prequantized: LHS=MXFP8, RHS=MXFP8",
        seed=52,
    ),
]

GRID_FP8_DTYPES = [
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        float8_dtype="float8_e5m2",
        description="FP8 E5M2: LHS=MXFP8, RHS=BF16",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        rhs_dtype=constants.MatrixPrecision.MXFP8_X4,
        float8_dtype="float8_e5m2",
        description="FP8 E5M2: LHS=MXFP8, RHS=MXFP8",
        seed=52,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.MXFP8,
        float8_dtype="float8_e5m2",
        description="FP8 E5M2: LHS=MXFP8, RHS=MXFP8",
        seed=52,
    ),
]

GRID_LNC = [
    config_helper.TestConfig(
        M=1024, K=1024, N=3008, run_with_lnc2=True, description="LNC2 enabled: M=1024, K=1024, N=3008", seed=52
    ),
    config_helper.TestConfig(
        M=1024, K=1024, N=2048, run_with_lnc2=False, description="LNC1: M=1024, K=1024, N=2048", seed=52
    ),
]

GRID_PACKED_SCALES = [
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        output_dtype=constants.MatrixPrecision.FP32,
        description="Packed scales: LHS=MXFP8",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
        fast_subset={1},
    ),
    config_helper.TestConfig(
        M=1056,
        K=1024,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.MXFP8,
        output_dtype=constants.MatrixPrecision.FP32,
        description="Packed scales: RHS=MXFP8, non divisible Mgi",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=2048,
        K=2048,
        N=2112,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.MXFP8,
        output_dtype=constants.MatrixPrecision.FP32,
        description="Packed scales: LHS=MXFP8, RHS=MXFP8, non divisible N",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
    ),
]

GRID_BF16_SCALE_PACKING = [
    config_helper.TestConfig(
        M=1024,
        K=2048,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        output_dtype=constants.MatrixPrecision.FP32,
        description="BF16 scale packing: LHS=BF16, RHS=BF16",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
    ),
    config_helper.TestConfig(
        M=1056,
        K=2048,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        output_dtype=constants.MatrixPrecision.FP32,
        description="BF16 scale packing: non divisible M",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
    ),
    config_helper.TestConfig(
        M=2048,
        K=2048,
        N=2112,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        output_dtype=constants.MatrixPrecision.FP32,
        description="BF16 scale packing: non divisible N",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
    ),
    config_helper.TestConfig(
        M=2048,
        K=640,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        output_dtype=constants.MatrixPrecision.FP32,
        description="BF16 scale packing: non divisible K",
        seed=52,
        tile_k=512,
        enable_scale_packing=True,
    ),
]

GRID_UNSWIZZLED = [
    config_helper.TestConfig(
        M=512,
        K=512,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Minimal example ",
        seed=52,
        spill_reload=False,
    ),
    config_helper.TestConfig(
        M=2048,
        K=2048,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=4,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Square Matrix",
        seed=52,
        fast_subset={2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=512,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled: larger matrices",
        run_with_lnc2=False,
        seed=52,
        spill_reload=False,
    ),
    config_helper.TestConfig(
        M=1024,
        K=512,
        N=2080,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        run_with_lnc2=False,
        description="N Not divisible",
        seed=52,
        spill_reload=False,
    ),
    config_helper.TestConfig(
        M=1216,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="M not divisible ",
        run_with_lnc2=False,
        seed=52,
        spill_reload=False,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="lhs not swizzled, rhs swizzled",
        seed=52,
        spill_reload=False,
        fast_subset={1, 2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=True,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="rhs not swizzled, lhs swizzled",
        seed=52,
        spill_reload=False,
        fast_subset={1, 2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.MXFP8,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=True,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="lhs pre quantized, rhs not swizzled",
        seed=52,
        spill_reload=False,
        fast_subset={1, 2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=512,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Spill reload",
        run_with_lnc2=False,
        seed=52,
        spill_reload=True,
    ),
    config_helper.TestConfig(
        M=1024,
        K=512,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Run with LNC2",
        run_with_lnc2=True,
        seed=52,
        spill_reload=False,
    ),
    config_helper.TestConfig(
        M=1024,
        K=512,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Scale Packing",
        run_with_lnc2=False,
        seed=52,
        spill_reload=False,
        enable_scale_packing=True,
    ),
]


GRID_UNSWIZZLED_K_DIV_128 = [
    # K=128 (minimum, no full 512-tiles)
    config_helper.TestConfig(
        M=512,
        K=128,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=128 (min, no full tiles)",
        seed=52,
        spill_reload=False,
        fast_subset={2},
    ),
    # K=256 (no full 512-tiles, remainder=256)
    config_helper.TestConfig(
        M=512,
        K=256,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=256 (no full tiles)",
        seed=52,
        spill_reload=False,
    ),
    # K=384 (no full 512-tiles, remainder=256+128)
    config_helper.TestConfig(
        M=512,
        K=384,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=384 (remainder 256+128)",
        seed=52,
        spill_reload=False,
    ),
    # K=640 (1 full 512-tile + 128 remainder)
    config_helper.TestConfig(
        M=512,
        K=640,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=640 (1 full + 128 remainder)",
        seed=52,
        spill_reload=False,
    ),
    # K=768 (1 full 512-tile + 256 remainder)
    config_helper.TestConfig(
        M=512,
        K=768,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=768 (1 full + 256 remainder)",
        seed=52,
        spill_reload=False,
    ),
    # K=896 (1 full 512-tile + 384 remainder = 256+128)
    config_helper.TestConfig(
        M=1024,
        K=896,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=896 (1 full + 256+128 remainder)",
        seed=52,
        spill_reload=False,
    ),
    # K=1152 (2 full 512-tiles + 128 remainder)
    config_helper.TestConfig(
        M=512,
        K=1152,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled K=1152 (2 full + 128 remainder)",
        seed=52,
        spill_reload=False,
    ),
    # Mixed: LHS unswizzled, RHS swizzled, K not div by 512
    config_helper.TestConfig(
        M=1024,
        K=512 + 384,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled LHS only, K=896",
        seed=52,
        spill_reload=False,
    ),
    # Non divisible M
    config_helper.TestConfig(
        M=1312,
        K=640,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=8,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled LHS only, K=640",
        seed=52,
        spill_reload=False,
        fast_subset={2},
    ),
    # Non divisible N
    config_helper.TestConfig(
        M=1024,
        K=640,
        N=2368,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled LHS only, K=640",
        seed=52,
        spill_reload=False,
        fast_subset={1},
    ),
    # Large M and N
    config_helper.TestConfig(
        M=4096,
        K=768,
        N=4096,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=16,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled LHS only, K=640",
        seed=52,
        spill_reload=False,
    ),
    # Large K
    config_helper.TestConfig(
        M=2048,
        K=4224,
        N=4096,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Unswizzled LHS only, K=640",
        seed=52,
        spill_reload=False,
        fast_subset={0, 1},
    ),
    # Spill reload
    config_helper.TestConfig(
        M=2048,
        K=4224,
        N=4096,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Spill reload",
        seed=52,
        spill_reload=True,
        fast_subset={0, 1},
    ),
    # Scale packing
    config_helper.TestConfig(
        M=1024,
        K=640,
        N=2368,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="Enable scale packing",
        enable_scale_packing=True,
        fast_subset={1},
    ),
]

GRID_PE_SWIZZLE = [
    config_helper.TestConfig(
        M=512,
        K=512,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="PE swizzle: minimal",
        seed=52,
        load_with_PE_swizzle=True,
        fast_subset={2},
    ),
    config_helper.TestConfig(
        M=2048,
        K=2048,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=4,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="PE swizzle: square matrix",
        seed=52,
        load_with_PE_swizzle=True,
        fast_subset={2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="PE swizzle: LHS only, RHS swizzled",
        seed=52,
        load_with_PE_swizzle=True,
    ),
    config_helper.TestConfig(
        M=1024,
        K=768,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="PE swizzle: K not divisible by 512",
        seed=52,
        load_with_PE_swizzle=True,
    ),
    config_helper.TestConfig(
        M=1024,
        K=384,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="PE swizzle: K=384",
        seed=52,
        load_with_PE_swizzle=True,
    ),
]

GRID_K_BY_F = [
    config_helper.TestConfig(
        M=512,
        K=512,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: both sides",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=True,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=2,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: LHS only, RHS swizzled",
        seed=52,
        lhs_is_f_by_k=False,
    ),
    config_helper.TestConfig(
        M=1024,
        K=768,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: K remainder 256",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # K%512=384
    config_helper.TestConfig(
        M=512,
        K=896,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: K%512=384",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # K%512=256
    config_helper.TestConfig(
        M=512,
        K=256,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: K=256 (no full tiles)",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # K%512=128
    config_helper.TestConfig(
        M=512,
        K=640,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: K%512=128",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # Large shape
    config_helper.TestConfig(
        M=4096,
        K=4096,
        N=3072,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=2,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=2,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="K-by-F: large 4096x4096x3072",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # NOTE: K-by-F currently requires F (M for LHS, N for RHS) to be a multiple of 512 (the F
    # load-tile size) — see the F%512 kernel_assert in matmul_mxfp8(). Non-512 F (e.g. F%128)
    # is not yet supported by the PE-transpose load and is rejected by that assert; it will be
    # enabled once the DMA gather-transpose API can mask partial F-tiles. All shapes here keep
    # M and N multiples of 512.
]


def get_output_dtype(conf):
    """Get NKI output dtype from test configuration.

    Args:
        conf: Test configuration object.

    Returns:
        nl.dtype: NKI dtype for output tensor (nl.float32 or nl.bfloat16).

    Raises:
        ValueError: If output dtype is not supported.
    """
    if conf.output_dtype == constants.MatrixPrecision.FP32:
        return nl.float32
    elif conf.output_dtype == constants.MatrixPrecision.BFLOAT16:
        return nl.bfloat16
    elif conf.output_dtype is None:
        return nl.bfloat16
    else:
        raise ValueError(f"Unsupported output dtype: {conf.output_dtype}")


def build_matmul_inputs(conf):
    """Build kernel inputs from test configuration.

    Generates random input tensors, applies swizzling (if configured), and optionally quantizes
    to MXFP8 format based on configuration.

    Args:
        conf: Test configuration object specifying matrix dimensions, dtypes,
            tile sizes, and other kernel parameters.

    Returns:
        dict: kernel_input arguments for kernel execution.
    """
    from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import _get_mx_max_exp

    lhs_fp32, rhs_fp32 = random_input_generator.get_random_inputs(
        {
            "shapes": (conf.M, conf.K, conf.N),
            "dists": conf.dists or ["normal", "normal"],
            "params": conf.params or [{}, {}],
        },
        conf.seed,
    )

    # Create swizzled versions. The quant scheme selects the interleave layout:
    # wrapX scatters a feature's K into four quarters; 1x32 packs four consecutive
    # K values per feature. Pre-quantized operands are quantized from this layout,
    # so it must match the scheme the kernel loads with.
    swizzle = matmul_utils.swizzle_tensor_1x32 if conf.quant_scheme == "1x32" else matmul_utils.swizzle_tensor
    lhs_swizzled = swizzle(lhs_fp32.numpy().astype(nl.bfloat16).T)
    rhs_swizzled = swizzle(rhs_fp32.numpy().astype(nl.bfloat16))

    # For kernel input, use swizzled or unswizzled based on config
    if conf.lhs_is_swizzled:
        lhs = lhs_swizzled
    elif not getattr(conf, 'lhs_is_f_by_k', True):
        # K-by-F: [K, M] layout
        lhs = lhs_fp32.numpy().astype(nl.bfloat16).T
    else:
        lhs = lhs_fp32.numpy().astype(nl.bfloat16)

    if conf.rhs_is_swizzled:
        rhs = rhs_swizzled
    elif not getattr(conf, 'rhs_is_f_by_k', True):
        # K-by-F: [K, N] layout
        rhs = rhs_fp32.numpy().astype(nl.bfloat16)
    else:
        rhs = rhs_fp32.numpy().astype(nl.bfloat16).T

    float8_dtype_x4 = conf.float8_dtype + "_x4" if not conf.float8_dtype.endswith("_x4") else conf.float8_dtype
    float8_dtype_non_x4 = (
        conf.float8_dtype if not conf.float8_dtype.endswith("_x4") else conf.float8_dtype.replace("_x4", "")
    )
    neuron_float8_dtype_x4 = getattr(dtype, float8_dtype_x4)
    neuron_float8_dtype_non_x4 = getattr(dtype, float8_dtype_non_x4)
    lhs_scales = None
    rhs_scales = None

    if conf.lhs_dtype == constants.MatrixPrecision.MXFP8 or conf.lhs_dtype == constants.MatrixPrecision.MXFP8_X4:
        lhs_quantized_data, lhs_quantized_scales_packed = mx_util.quantize_mx_golden(
            lhs_swizzled, neuron_float8_dtype_x4, custom_mx_max_exp=_get_mx_max_exp
        )
        if conf.enable_scale_packing:
            lhs_quantized_scales = generate_golden_packed_scales(lhs_quantized_scales_packed, conf.K, conf.M, Q_TILE_K)
        else:
            lhs_quantized_scales = matmul_utils.resize_scales_compact_to_oversized_2d(lhs_quantized_scales_packed)
        lhs = lhs_quantized_data
        lhs_scales = lhs_quantized_scales
        if conf.lhs_dtype == constants.MatrixPrecision.MXFP8:
            lhs = lhs.view(neuron_float8_dtype_non_x4)

    if conf.rhs_dtype == constants.MatrixPrecision.MXFP8 or conf.rhs_dtype == constants.MatrixPrecision.MXFP8_X4:
        rhs_quantized_data, rhs_quantized_scales_packed = mx_util.quantize_mx_golden(
            rhs_swizzled, neuron_float8_dtype_x4, custom_mx_max_exp=_get_mx_max_exp
        )
        if conf.enable_scale_packing:
            rhs_quantized_scales = generate_golden_packed_scales(rhs_quantized_scales_packed, conf.K, conf.N, Q_TILE_K)
        else:
            rhs_quantized_scales = matmul_utils.resize_scales_compact_to_oversized_2d(rhs_quantized_scales_packed)
        rhs = rhs_quantized_data
        rhs_scales = rhs_quantized_scales
        if conf.rhs_dtype == constants.MatrixPrecision.MXFP8:
            rhs = rhs.view(neuron_float8_dtype_non_x4)

    return {
        "lhs": lhs,
        "rhs": rhs,
        "TILES_IN_BLOCK_M": conf.TILES_IN_BLOCK_M,
        "TILES_IN_BLOCK_N": conf.TILES_IN_BLOCK_N,
        "TILES_IN_BLOCK_K": conf.TILES_IN_BLOCK_K,
        "TILES_IN_LOAD_M": conf.TILES_IN_LOAD_M,
        "TILES_IN_LOAD_N": conf.TILES_IN_LOAD_N,
        "lhs_matmul_tile_shape_logical": (conf.tile_k, conf.tile_m),
        "rhs_matmul_tile_shape_logical": (conf.tile_k, conf.tile_n),
        "block_loop_order": conf.block_loop_order,
        "tile_loop_order": conf.tile_loop_order,
        "output_dtype": get_output_dtype(conf),
        "run_with_lnc2": conf.run_with_lnc2,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "float8_dtype": conf.float8_dtype,
        "use_scale_packing": conf.enable_scale_packing,
        "spill_reload": conf.spill_reload,
        "lhs_is_swizzled": conf.lhs_is_swizzled,
        "rhs_is_swizzled": conf.rhs_is_swizzled,
        **({"load_with_PE_swizzle": True} if conf.load_with_PE_swizzle else {}),
        **(
            {"lhs_is_f_by_k": False}
            if not conf.lhs_is_swizzled and getattr(conf, 'lhs_is_f_by_k', None) is False
            else {}
        ),
        **(
            {"rhs_is_f_by_k": False}
            if not conf.rhs_is_swizzled and getattr(conf, 'rhs_is_f_by_k', None) is False
            else {}
        ),
        **({"lnc_2_shard_rhs": conf.lnc_2_shard_rhs} if conf.lnc_2_shard_rhs is not None else {}),
        "quant_scheme": conf.quant_scheme,
        "enable_psum_copy_in": conf.enable_psum_copy_in,
    }


def generate_chain(dimension):
    """Generate test parameter chains for a specific matrix dimension.

    Creates random dimension sizes and corresponding tile configurations for
    parametrized testing.

    Args:
        dimension: Dimension to generate chain for ("M", "N", or "K").

    Returns:
        list: List of tuples containing dimension-specific test parameters:
            - For "M": (M, tile_m, TILES_IN_BLOCK_M, TILES_IN_LOAD_M)
            - For "N": (N, tile_n, TILES_IN_BLOCK_N, TILES_IN_LOAD_N)
            - For "K": (K, tile_k, TILES_IN_BLOCK_K)

    Raises:
        ValueError: If dimension is not "M", "N", or "K".
    """
    chain = []
    if dimension == "M":
        M_VALUES = random.sample(range(128, 8193, 32), SWEEP_NUM_VALUES)
        for m_value in M_VALUES:
            dummy_config = config_helper.TestConfig(M=m_value, N=1024, K=1024)
            temp = dummy_config._generate_m_dimension_tile_configs()
            temp = [(m_value, *each) for each in temp]
            chain += random.sample(temp, 1)

    elif dimension == "N":
        N_VALUES = random.sample(range(128, 8193, 32), SWEEP_NUM_VALUES)
        for n_value in N_VALUES:
            dummy_config = config_helper.TestConfig(M=1024, N=n_value, K=1024)
            temp = dummy_config._generate_n_dimension_tile_configs()
            temp = [(n_value, *each) for each in temp]
            chain += random.sample(temp, 1)

    elif dimension == "K":
        K_VALUES = random.sample(range(128, 8193, 128), SWEEP_NUM_VALUES)
        for k_value in K_VALUES:
            dummy_config = config_helper.TestConfig(M=1024, N=1024, K=k_value)
            temp = dummy_config._generate_k_dimension_tile_configs()
            temp = [(k_value, *each) for each in temp]
            chain += random.sample(temp, 1)
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    return chain


def filter_illegal_combinations(
    m_chain,
    n_chain,
    k_chain,
    lhs_is_swizzled=None,
    rhs_is_swizzled=None,
    lnc_2_shard_rhs=None,
    run_with_lnc2=None,
    lhs_dtype=None,
    rhs_dtype=None,
    enable_scale_packing=None,
    output_dtype=None,
    tile_loop_order=None,
    block_loop_order=None,
    float8_dtype='float8_e4m3fn',
    lhs_dist=None,
    rhs_dist=None,
    spill_reload=None,
) -> coverage_parametrized_tests.FilterResult:
    """Filter out parameter combinations that exceed SBUF capacity.

    Validates that the test configuration fits within Trainium SBUF memory
    constraints before test execution. Supports partial combinations where
    some parameters may be None during incremental coverage sweeps.

    Args:
        m_chain: M dimension parameters (M, tile_m, TILES_IN_BLOCK_M, TILES_IN_LOAD_M).
        n_chain: N dimension parameters (N, tile_n, TILES_IN_BLOCK_N, TILES_IN_LOAD_N).
        k_chain: K dimension parameters (K, tile_k, TILES_IN_BLOCK_K).
        lhs_is_swizzled (bool, optional): Whether LHS matrix uses swizzled layout.
        rhs_is_swizzled (bool, optional): Whether RHS matrix uses swizzled layout.
        lhs_dtype (optional): Left-hand side matrix precision.
        rhs_dtype (optional): Right-hand side matrix precision.
        output_dtype (optional): Output matrix precision.
        tile_loop_order (str, optional): Tile loop iteration order (e.g., "mnk").
        block_loop_order (str, optional): Block loop iteration order (e.g., "mnk").
        float8_dtype (str): Float8 dtype string ("float8_e4m3fn" or "float8_e5m2").
        lhs_dist (optional): LHS distribution parameters (distribution_name, json_params).
        rhs_dist (optional): RHS distribution parameters (distribution_name, json_params).
        run_with_lnc2 (bool, optional): Whether to run with LNC2 enabled.

    Returns:
        FilterResult: VALID if configuration fits in SBUF or combination is partial, INVALID otherwise.
    """
    M, tile_m, TILES_IN_BLOCK_M, TILES_IN_LOAD_M = m_chain
    N, tile_n, TILES_IN_BLOCK_N, TILES_IN_LOAD_N = n_chain
    K, tile_k, TILES_IN_BLOCK_K = k_chain

    # partial combination
    if lhs_is_swizzled is None or rhs_is_swizzled is None:
        return coverage_parametrized_tests.FilterResult.VALID

    # swizzling shape constraints
    valid_for_not_swizzled = True
    if not lhs_is_swizzled:
        valid_for_not_swizzled = (
            valid_for_not_swizzled and K % 128 == 0 and tile_m == 128 and TILES_IN_LOAD_M == 4 and tile_k == 512
        )
    if not rhs_is_swizzled:
        valid_for_not_swizzled = (
            valid_for_not_swizzled and K % 128 == 0 and tile_n == 512 and TILES_IN_LOAD_N == 1 and tile_k == 512
        )
    if not valid_for_not_swizzled:
        return coverage_parametrized_tests.FilterResult.INVALID

    # LNC2 sharding requires at least 2 blocks in the sharded dimension
    if lnc_2_shard_rhs is not None and run_with_lnc2 is not None and run_with_lnc2:
        if not lnc_2_shard_rhs:
            num_blocks_in_m = M // (tile_m * TILES_IN_BLOCK_M) if (tile_m * TILES_IN_BLOCK_M) > 0 else 0
            if num_blocks_in_m < 2:
                return coverage_parametrized_tests.FilterResult.INVALID
        else:
            num_blocks_in_n = N // (tile_n * TILES_IN_BLOCK_N) if (tile_n * TILES_IN_BLOCK_N) > 0 else 0
            if num_blocks_in_n < 2:
                return coverage_parametrized_tests.FilterResult.INVALID

    # Prune early once lhs_dtype, rhs_dtype, and enable_scale_packing are known
    if lhs_dtype is not None and rhs_dtype is not None and enable_scale_packing is not None:
        """
        Pre-quantized MXFP8 with packed scales requires tile_k=512 (tile_k=Q_TILE_K=128)
        because the packed scales format uses Q_TILE_K-sized tile indexing that doesn't
        align with smaller physical tile boundaries in the matmul instruction.
        # TODO(zolcsaki): Clarify tile_k naming — tile_k is the logical tile size parameter
        # passed to the test (e.g., 512), while Q_TILE_K (128) is the physical tile size
        # used by the quantization hardware. They are related by INTERLEAVE_FACTOR.
        """
        if enable_scale_packing and tile_k < 512:
            is_lhs_prequantized = lhs_dtype in (constants.MatrixPrecision.MXFP8, constants.MatrixPrecision.MXFP8_X4)
            is_rhs_prequantized = rhs_dtype in (constants.MatrixPrecision.MXFP8, constants.MatrixPrecision.MXFP8_X4)
            if is_lhs_prequantized or is_rhs_prequantized:
                return coverage_parametrized_tests.FilterResult.INVALID

    if (
        lhs_dtype is None
        or rhs_dtype is None
        or output_dtype is None
        or tile_loop_order is None
        or block_loop_order is None
        or run_with_lnc2 is None
    ):
        return coverage_parametrized_tests.FilterResult.VALID

    config = config_helper.TestConfig(
        M=M,
        N=N,
        K=K,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        tile_loop_order=tile_loop_order,
        block_loop_order=block_loop_order,
        run_with_lnc2=run_with_lnc2,
        float8_dtype=float8_dtype,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        output_dtype=output_dtype,
    )

    return (
        coverage_parametrized_tests.FilterResult.VALID
        if config.fits_in_sbuf()
        else coverage_parametrized_tests.FilterResult.INVALID
    )


def _mxfp8_comparator(conf, output_dtype, gpu_golden_enabled=False):
    """Return a custom_comparator closure for MXFP8 matmul validation.

    The returned callable receives MXFP8 golden from torch_ref and wraps it
    in a CustomValidator that uses check_correctness as the primary validation.

    When gpu_golden_enabled=True, the validator also performs a supplementary
    GPU threshold check comparing kernel output against FP32 golden. The FP32
    golden is regenerated from conf.seed (same as build_matmul_inputs) because
    FP32 precision is lost during BF16 conversion. This supplementary check
    catches quality regressions in quantization error.
    """

    gpu_golden_threshold = 0
    if gpu_golden_enabled:
        conf_hash = matmul_utils.dict_hash(conf.get_input_gen_config())
        gpu_golden_threshold = GPU_THRESHOLDS.get(conf_hash, 0)

    def comparator(golden_dict, output_tensors):
        golden = golden_dict["out"]

        class _MatmulValidator(common_dataclasses.CustomValidator):
            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                reshaped = inference_output.view(dtype=output_dtype).astype(output_dtype).reshape(golden.shape)

                # Primary: CPU MXFP8 golden validation (uses torch_ref output)
                passed, metrics = matmul_utils.check_correctness(reshaped, golden.astype(output_dtype))
                if not passed:
                    self._print_with_log(f"NKI_Ker[0, :5] {reshaped[0, :5]}")
                    self._print_with_log(f"Golden[0, :5] {golden[0, :5]}")
                    self._print_with_log("CPU validation failed")
                    self._print_with_log(f"CPU check statistics: {metrics}")
                    return False

                # Supplementary: GPU FP32 threshold validation
                if gpu_golden_enabled:
                    lhs_fp32, rhs_fp32 = random_input_generator.get_random_inputs(
                        {
                            "shapes": (conf.M, conf.K, conf.N),
                            "dists": conf.dists or ["normal", "normal"],
                            "params": conf.params or [{}, {}],
                        },
                        conf.seed,
                    )
                    fp32_golden = lhs_fp32.numpy() @ rhs_fp32.numpy()
                    trn_threshold = np.linalg.norm(
                        fp32_golden.astype(np.float64) - reshaped.astype(np.float64)
                    ) / np.linalg.norm(fp32_golden.astype(np.float64))
                    gpu_passed = trn_threshold <= gpu_golden_threshold + 3e-3
                    if not gpu_passed:
                        self._print_with_log(f"NKI_Ker[0, :5] {reshaped[0, :5]}")
                        self._print_with_log(f"CPU_FP32[0, :5] {fp32_golden[0, :5]}")
                        self._print_with_log("GPU validation failed")
                        self._print_with_log(f"TRN threshold: {trn_threshold}, GPU threshold: {gpu_golden_threshold}")
                        self._print_with_log(f"Config: {conf.__repr__()}")
                        return False

                return True

        return {
            "out": common_dataclasses.CustomValidatorWithOutputTensorData(
                validator=_MatmulValidator,
                output_ndarray=np.ndarray(shape=(conf.M, conf.N), dtype=output_dtype),
            )
        }

    return comparator


_ABBREVS = {
    "lhs_is_swizzled": "lw",
    "rhs_is_swizzled": "rw",
    "lhs_dtype": "ld",
    "rhs_dtype": "rd",
    "output_dtype": "od",
    "tile_loop_order": "tl",
    "block_loop_order": "bl",
    "float8_dtype": "f",
    "lhs_dist": "x",
    "rhs_dist": "y",
    "m_chain": "m",
    "n_chain": "n",
    "k_chain": "k",
    "spill_reload": "s",
    "enable_scale_packing": "p",
    "enable_psum_copy_in": "c",
    "lnc_2_shard_rhs": "h",
    "run_with_lnc2": "l",
}

# TODO: Add quant_scheme=["wrapX", "1x32"] to test_matmul_mxfp8_sweep once pre-quantized/pre-swizzled
# inputs properly bypass the 1x32 quantize path in the load pipeline. Currently, mixed configs
# (e.g., one operand pre-quantized wrapX + other operand BF16 with 1x32) cause OOB compilation errors.

GRID_1X32 = [
    config_helper.TestConfig(
        M=512,
        K=512,
        N=512,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: minimal",
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=2048,
        K=2048,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=4,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: square matrix",
        seed=52,
        quant_scheme="1x32",
        fast_subset={2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=1536,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=4,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: non-square",
        seed=52,
        quant_scheme="1x32",
    ),
    config_helper.TestConfig(
        M=1216,
        K=1024,
        N=2048,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=2,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: non-divisible M",
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=2080,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=2,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: non-divisible N",
        run_with_lnc2=False,
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=1024,
        K=768,
        N=1024,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=1,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: non-divisible K",
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=640,
        K=1536,
        N=3008,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=3,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: all dims non-divisible",
        run_with_lnc2=False,
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
    config_helper.TestConfig(
        M=4096,
        K=4096,
        N=1536,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_K=2,
        TILES_IN_BLOCK_M=16,
        TILES_IN_BLOCK_N=3,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        description="1x32: large shape 4096x4096x1536",
        seed=52,
        quant_scheme="1x32",
        fast_subset={0, 1, 2},
    ),
]


@pytest_test_metadata(name="Matmul MXFP8")
@pytest_marks(["matmul_mxfp8", "mx", "mxfp8"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestMatmulMxfp8GenericKernel:
    def run_matmul_mxfp8_generic_test(
        self,
        test_manager: test_orchestrator.Orchestrator,
        compiler_args: common_dataclasses.CompilerArgs,
        conf: config_helper.TestConfig,
        is_negative_test: bool = False,
        gpu_golden_enabled: bool = True,
    ):
        """Execute MXFP8 matrix multiplication test with specified configuration.

        Args:
            test_manager: Test orchestrator for kernel execution and validation.
            compiler_args: Compiler configuration arguments.
            conf: Test configuration specifying matrix dimensions and parameters.
            is_negative_test: If True, expect the test to fail. Default is False.
            gpu_golden_enabled: If True, validate against GPU golden threshold. Default is True.
        """

        if os.environ.get('TEST_COLLECTION_BENCHMARK') == '1':
            pytest.skip("Benchmark mode - skipping test execution")

        test_manager.collector.set_kernel_params(conf.to_metrics_dict())
        output_dtype = get_output_dtype(conf)

        def input_generator(test_config):
            return build_matmul_inputs(conf)

        def output_tensors(kernel_input):
            return {"out": np.zeros((conf.M, conf.N), dtype=output_dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_kernel.matmul_mxfp8,
            torch_ref=matmul_mxfp8_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            is_negative_test=is_negative_test,
            custom_comparator=_mxfp8_comparator(conf, output_dtype, gpu_golden_enabled),
        )

    @pytest.mark.parametrize("conf", populate_tests(GRID))
    def test_matmul_mxfp8_base_grid(self, test_manager, conf, platform_target):
        """Test basic MXFP8 matrix multiplication functionality.

        Validates kernel correctness on a base grid of matrix sizes including
        small, medium, large, and non-square configurations.
        """
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    @pytest.mark.parametrize("conf", populate_tests(GRID_LARGE))
    def test_matmul_mxfp8_base_grid_large(self, test_manager, conf, platform_target):
        """Test MXFP8 matrix multiplication with large matrix configurations."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    @pytest.mark.parametrize("conf", populate_tests(GRID_NON_DIVISIBLE_FAST))
    def test_matmul_mxfp8_non_divisible_grid_fast(self, test_manager, conf, platform_target):
        """Fast subset of non-divisible grid to cover remainder/masking path (KTK-118)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf)

    @pytest.mark.parametrize(
        "conf",
        populate_tests(
            GRID_SWEEP_M + GRID_SWEEP_N + GRID_SWEEP_K + GRID_ALL_NON_DIVISIBLE + GRID_PARTIAL_NON_DIVISIBLE
        ),
    )
    def test_matmul_mxfp8_non_divisible_grid(self, test_manager, conf, platform_target):
        """Test MXFP8 matrix multiplication with non-tile-aligned dimensions.

        Validates masking support for matrix dimensions that are not divisible
        by tile sizes, sweeping M, N, and K dimensions independently and in combination.
        """
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf)

    @pytest.mark.parametrize("conf", populate_tests(GRID_PREQUANTIZED + GRID_FP8_DTYPES))
    def test_matmul_mxfp8_prequantized_grid(self, test_manager, conf, platform_target):
        """Quick validation tests for prequantized inputs."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf)

    @pytest.mark.parametrize("conf", populate_tests(GRID_LNC))
    def test_matmul_mxfp8_lnc_grid(self, test_manager, conf, platform_target):
        """Quick validation tests for different LNC degree."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf)

    @pytest.mark.parametrize("conf", populate_tests(GRID_PACKED_SCALES))
    def test_matmul_mxfp8_packed_scales(self, test_manager, conf, platform_target):
        """Test matmul with pre-quantized packed scales inputs."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.parametrize("conf", populate_tests(GRID_BF16_SCALE_PACKING))
    def test_matmul_mxfp8_bf16_scale_packing(self, test_manager, conf, platform_target):
        """Test matmul with bf16 inputs and scale packing enabled."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    @pytest.mark.parametrize("conf", populate_tests(GRID_UNSWIZZLED))
    def test_matmul_mxfp8_unswizzled(self, test_manager, conf, platform_target):
        """Test matmul with unswizzled BF16 inputs in [F, K] format."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    @pytest.mark.parametrize("conf", populate_tests(GRID_UNSWIZZLED_K_DIV_128))
    def test_matmul_mxfp8_unswizzled_k_div_128(self, test_manager, conf, platform_target):
        """Test matmul with unswizzled BF16 inputs where K is divisible by 128 but not 512."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.parametrize("conf", populate_tests(GRID_PE_SWIZZLE))
    def test_matmul_mxfp8_pe_swizzle(self, test_manager, conf, platform_target):
        """Test matmul with unswizzled BF16 inputs using PE swizzle loading instead of DGT."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.fast
    @pytest.mark.parametrize("conf", populate_tests(GRID_K_BY_F))
    def test_matmul_mxfp8_k_by_f(self, test_manager, conf, platform_target):
        """Test matmul with K-by-F unswizzled BF16 inputs (PE swizzle auto-forced)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.parametrize(
        "conf",
        populate_tests(model_config_reader.qwen3_8b_tp16 + model_config_reader.qwen3_8b_tp4),
    )
    def test_matmul_mxfp8_qwen3_8b_grid(self, test_manager, conf, platform_target):
        """Quick validation tests for different LNC degree."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    @pytest.mark.parametrize(
        "conf",
        populate_tests(model_config_reader.qwen3_235b_cp16_tp4 + model_config_reader.qwen3_235b_cp4_tp4),
    )
    def test_matmul_mxfp8_qwen3_235b_grid(self, test_manager, conf, platform_target):
        """Quick validation tests for different LNC degree."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=True)

    def test_matmul_mxfp8_gpt_oss_kv_proj_defaults(self, test_manager, platform_target):
        """GPT-OSS-20B TP4 K/V projection (M=2048, K=2880, N=128) with all defaults."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        np.random.seed(42)
        conf = config_helper.TestConfig(M=2048, K=2880, N=128)
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (512, 512, 512),
            (2048, 2048, 2048),
            (640, 1600, 3008),
            (1312, 2560, 2880),
            (4096, 4096, 1536),
        ],
        ids=["small_square", "medium_square", "non_divisible_all", "non_divisible_M", "qwen3_qkv_proj"],
    )
    def test_matmul_mxfp8_shape_only_defaults(self, test_manager, platform_target, M, K, N):
        """Test matmul with only shapes specified, letting auto_generate_default configure everything."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        np.random.seed(42)
        conf = config_helper.TestConfig(M=M, K=K, N=N)
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    @pytest.mark.coverage_parametrize(
        m_chain=generate_chain("M"),
        n_chain=generate_chain("N"),
        k_chain=generate_chain("K"),
        lhs_is_swizzled=[True, False],
        rhs_is_swizzled=[True, False],
        lnc_2_shard_rhs=[True, False],
        run_with_lnc2=[True, False],
        lhs_dtype=[
            constants.MatrixPrecision.BFLOAT16,
            constants.MatrixPrecision.MXFP8,
            constants.MatrixPrecision.MXFP8_X4,
        ],
        rhs_dtype=[
            constants.MatrixPrecision.BFLOAT16,
            constants.MatrixPrecision.MXFP8,
            constants.MatrixPrecision.MXFP8_X4,
        ],
        output_dtype=[constants.MatrixPrecision.FP32, constants.MatrixPrecision.BFLOAT16],
        tile_loop_order=['mnk'],
        block_loop_order=['mnk'],
        float8_dtype=['float8_e4m3fn', 'float8_e5m2'],
        lhs_dist=get_dists(num=SWEEP_NUM_VALUES, edge_rate=0.3),
        rhs_dist=get_dists(num=SWEEP_NUM_VALUES, edge_rate=0.3),
        spill_reload=[True, False],
        enable_scale_packing=[True, False],
        enable_psum_copy_in=[True, False],
        filter=filter_illegal_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,  # TODO: Fix assertions
        enable_invalid_combination_tests=False,  # TODO: Fix assertions
        abbrev=_ABBREVS,
    )
    def test_matmul_mxfp8_sweep(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target,
        m_chain,
        n_chain,
        k_chain,
        run_with_lnc2,
        tile_loop_order,
        block_loop_order,
        float8_dtype,
        lhs_dtype,
        rhs_dtype,
        enable_scale_packing,
        output_dtype,
        lhs_dist,
        rhs_dist,
        is_negative_test_case,
        lhs_is_swizzled,
        rhs_is_swizzled,
        spill_reload,
        lnc_2_shard_rhs,
        enable_psum_copy_in,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        M, tile_m, TILES_IN_BLOCK_M, TILES_IN_LOAD_M = m_chain
        N, tile_n, TILES_IN_BLOCK_N, TILES_IN_LOAD_N = n_chain
        K, tile_k, TILES_IN_BLOCK_K = k_chain
        dists = lhs_dist[0], rhs_dist[0]
        params = json.loads(lhs_dist[1]), json.loads(rhs_dist[1])
        conf = config_helper.TestConfig(
            M=M,
            N=N,
            K=K,
            TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=TILES_IN_LOAD_N,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            tile_loop_order=tile_loop_order,
            block_loop_order=block_loop_order,
            run_with_lnc2=run_with_lnc2,
            float8_dtype=float8_dtype,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            output_dtype=output_dtype,
            dists=dists,
            params=params,
            lhs_is_swizzled=lhs_is_swizzled,
            rhs_is_swizzled=rhs_is_swizzled,
            spill_reload=spill_reload,
            enable_scale_packing=enable_scale_packing,
            lnc_2_shard_rhs=lnc_2_shard_rhs,
        )
        conf.enable_psum_copy_in = enable_psum_copy_in
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(
            test_manager, compiler_args, conf, is_negative_test=is_negative_test_case, gpu_golden_enabled=False
        )

    @pytest.mark.parametrize("conf", populate_tests(GRID_1X32))
    def test_matmul_mxfp8_1x32(self, test_manager, conf, platform_target):
        """Test matmul with 1x32 quantization scheme using FP32 reinterpret PE swizzle."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)

    def test_matmul_mxfp8_4096x4096x1536_pe_swizzle_1x32_spill_reload(self, test_manager, platform_target):
        """Test 4096x4096x1536 PE swizzle with 1x32 quantization + spill_reload."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        np.random.seed(42)
        conf = config_helper.TestConfig(
            M=4096,
            K=4096,
            N=1536,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_is_swizzled=False,
            rhs_is_swizzled=False,
            tile_m=128,
            tile_n=512,
            tile_k=512,
            TILES_IN_BLOCK_K=2,
            TILES_IN_BLOCK_M=16,
            TILES_IN_BLOCK_N=3,
            TILES_IN_LOAD_M=4,
            TILES_IN_LOAD_N=1,
            quant_scheme="1x32",
            spill_reload=True,
        )
        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )
        self.run_matmul_mxfp8_generic_test(test_manager, compiler_args, conf, gpu_golden_enabled=False)
