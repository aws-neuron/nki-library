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

"""Integration tests for MXFP8 MoE backward pass with check_correctness validation.

Uses a three-metric custom validator (cosine similarity, normalized Euclidean distance,
allclose with scaled atol) matching the MLP MXFP8 pattern.
"""

import functools
import math
import random
from dataclasses import replace
from typing import final

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import get_tile_sizes
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import AffinityOption, ClampLimits, ShardOption
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import blockwise_mm_bwd_mxfp8
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8_torch import (
    blockwise_mm_bwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import (
    MXFP8MOEBwdConfig,
    QuantScheme,
    SwizzleMode,
    TransposeMode,
)

from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import (
    build_mxfp8_moe_bwd_inputs,
)

# Shared fwd/bwd validated-suite helpers: correctness check, comparator factory,
# param-marking, abbreviations. See mxfp8_moe_validated_common.
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    ABBREVS as _ABBREVS,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    PARAM_NAMES,
    make_moe_validated_comparator,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    build_params as _build_params_common,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    clamp_tiles as _clamp_tiles,
)
from test.integration.nkilib.experimental.moe_mxfp8.tuned_blocking import (
    MAX_DGT_LOAD_WIDTH,
    get_shape_tuned_config,
    tune_block_and_load_tiles,
)
from test.utils import common_dataclasses
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16

# ============================================================================
# Blocking params helper (backward-specific: 4 phases)
# ============================================================================


def _compute_moe_bwd_config(
    H: int,
    B: int,
    I_TP: int,
    tiles_m: int,
    tiles_n: int,
    tiles_k: int,
    run_with_lnc2: bool = True,
    spill_reload: bool = False,
    use_scale_packing: bool = False,
    single_expert_dense: bool = False,
) -> MXFP8MOEBwdConfig:
    """Compute a valid MXFP8MOEBwdConfig (per-phase blocking) for the given shape dimensions.

    Args:
        tiles_m: Desired TILES_IN_BLOCK_M (1-8 range).
        tiles_n: Desired TILES_IN_BLOCK_N (1-8 range).
        tiles_k: Desired TILES_IN_BLOCK_K (1-8 range).
    """
    num_shards = 2 if run_with_lnc2 else 1
    I_TP_PER_SHARD = I_TP // num_shards
    H_PER_SHARD = H // num_shards

    def phase_tiles(M, K, N):
        if H == 384:
            return get_tile_sizes(512, 512, 512)
        return get_tile_sizes(K, M, N)

    # Phase 1: output_grad[B, H] @ W_down[H, I_TP/shard]
    p1_tiles = phase_tiles(B, H, I_TP_PER_SHARD)
    p1_num_b_tiles = math.ceil(B / p1_tiles["tile_m"])
    p1_num_i_tiles = math.ceil(I_TP_PER_SHARD / p1_tiles["tile_n"])
    p1_num_k_tiles = math.ceil(H / p1_tiles["l_tile_k"])
    p1_block_m = _clamp_tiles(tiles_m, max(1, p1_num_b_tiles // 4))
    p1_block_n = _clamp_tiles(tiles_n, p1_num_i_tiles)
    p1_block_m, p1_load_m = tune_block_and_load_tiles(p1_block_m, p1_tiles["tile_m"])
    p1_block_n, p1_load_n = tune_block_and_load_tiles(p1_block_n, p1_tiles["tile_n"])
    phase1 = MatmulMxfp8KernelConfig(
        M=B,
        K=H,
        N=I_TP_PER_SHARD,
        tile_m=p1_tiles["tile_m"],
        tile_k=p1_tiles["l_tile_k"],
        tile_n=p1_tiles["tile_n"],
        TILES_IN_BLOCK_M=p1_block_m,
        TILES_IN_BLOCK_N=p1_block_n,
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p1_num_k_tiles),
        TILES_IN_LOAD_M=p1_load_m,
        TILES_IN_LOAD_N=p1_load_n,
        quant_scheme=QuantScheme.WRAPX,
        spill_reload=spill_reload,
        enable_scale_packing=use_scale_packing,
    )

    # Phase 2: d_gate_up[B, 2*I_TP] @ W_gate_up[2*I_TP, H/shard]
    p2_tiles = phase_tiles(B, 2 * I_TP, H_PER_SHARD)
    p2_num_b_tiles = math.ceil(B / p2_tiles["tile_m"])
    p2_num_h_tiles = math.ceil(H_PER_SHARD / p2_tiles["tile_n"])
    p2_num_k_tiles = math.ceil((2 * I_TP) / p2_tiles["l_tile_k"])
    p2_block_m = _clamp_tiles(tiles_m, max(1, p2_num_b_tiles // 4))
    p2_block_n = _clamp_tiles(tiles_n, p2_num_h_tiles)
    p2_block_m, p2_load_m = tune_block_and_load_tiles(p2_block_m, p2_tiles["tile_m"])
    p2_block_n, p2_load_n = tune_block_and_load_tiles(p2_block_n, p2_tiles["tile_n"])
    phase2 = MatmulMxfp8KernelConfig(
        M=B,
        K=2 * I_TP,
        N=H_PER_SHARD,
        tile_m=p2_tiles["tile_m"],
        tile_k=p2_tiles["l_tile_k"],
        tile_n=p2_tiles["tile_n"],
        TILES_IN_BLOCK_M=p2_block_m,
        TILES_IN_BLOCK_N=p2_block_n,
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p2_num_k_tiles),
        TILES_IN_LOAD_M=p2_load_m,
        TILES_IN_LOAD_N=p2_load_n,
        quant_scheme=QuantScheme.WRAPX,
        spill_reload=spill_reload,
        enable_scale_packing=use_scale_packing,
    )

    # Phase 3: d_gate_up_T[I_TP, B] @ hidden_states_T[B, H/shard]
    p3_m = 2 * I_TP if single_expert_dense else I_TP
    p3_tiles = phase_tiles(p3_m, B, H_PER_SHARD)
    p3_num_i_tiles = math.ceil(p3_m / p3_tiles["tile_m"])
    p3_num_h_tiles = math.ceil(H_PER_SHARD / p3_tiles["tile_n"])
    p3_num_k_tiles = math.ceil(B / p3_tiles["l_tile_k"])
    p3_block_n = _clamp_tiles(tiles_n, p3_num_h_tiles)
    if (H, B, I_TP, tiles_m, tiles_n, tiles_k) == (5120, 4096, 256, 4, 8, 8):
        p3_block_n = min(p3_block_n, 4)
    p3_block_m = _clamp_tiles(tiles_m, max(1, p3_num_i_tiles // 4))
    p3_block_m, p3_load_m = tune_block_and_load_tiles(p3_block_m, p3_tiles["tile_m"])
    p3_block_n, p3_load_n = tune_block_and_load_tiles(p3_block_n, p3_tiles["tile_n"])
    phase3 = MatmulMxfp8KernelConfig(
        M=p3_m,
        K=B,
        N=H_PER_SHARD,
        tile_m=p3_tiles["tile_m"],
        tile_k=p3_tiles["l_tile_k"],
        tile_n=p3_tiles["tile_n"],
        TILES_IN_BLOCK_M=p3_block_m,
        TILES_IN_BLOCK_N=p3_block_n,
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p3_num_k_tiles),
        TILES_IN_LOAD_M=p3_load_m,
        TILES_IN_LOAD_N=p3_load_n,
        quant_scheme=QuantScheme.WRAPX,
        spill_reload=spill_reload,
        enable_scale_packing=use_scale_packing,
    )

    # Phase 4: output_grad_T[H/shard, B] @ scaled_intermediate_T[B, I_TP]
    p4_tiles = phase_tiles(H_PER_SHARD, B, I_TP)
    p4_num_h_tiles = math.ceil(H_PER_SHARD / p4_tiles["tile_m"])
    p4_num_i_tiles = math.ceil(I_TP / p4_tiles["tile_n"])
    p4_num_k_tiles = math.ceil(B / p4_tiles["l_tile_k"])
    p4_block_m = _clamp_tiles(tiles_m, max(1, p4_num_h_tiles // 4))
    p4_block_n = _clamp_tiles(tiles_n, p4_num_i_tiles)
    p4_block_m, p4_load_m = tune_block_and_load_tiles(p4_block_m, p4_tiles["tile_m"])
    p4_block_n, p4_load_n = tune_block_and_load_tiles(p4_block_n, p4_tiles["tile_n"])
    phase4 = MatmulMxfp8KernelConfig(
        M=H_PER_SHARD,
        K=B,
        N=I_TP,
        tile_m=p4_tiles["tile_m"],
        tile_k=p4_tiles["l_tile_k"],
        tile_n=p4_tiles["tile_n"],
        TILES_IN_BLOCK_M=p4_block_m,
        TILES_IN_BLOCK_N=p4_block_n,
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p4_num_k_tiles),
        TILES_IN_LOAD_M=p4_load_m,
        TILES_IN_LOAD_N=p4_load_n,
        quant_scheme=QuantScheme.WRAPX,
        spill_reload=spill_reload,
        enable_scale_packing=use_scale_packing,
    )

    return MXFP8MOEBwdConfig(phase1_config=phase1, phase2_config=phase2, phase3_config=phase3, phase4_config=phase4)


def _moe_bwd_config_from_tuned(tuned_config, H, B, I_TP, run_with_lnc2=True, single_expert_dense=False):
    """Fill in the per-phase (M,K,N) on a tuned MXFP8MOEBwdConfig (see tuned_blocking.py), using
    the same per-phase mapping as _compute_moe_bwd_config. Lets the test drive the shape-tuned
    blocking as an explicit config."""
    num_shards = 2 if run_with_lnc2 else 1
    I_TP_PER_SHARD = I_TP // num_shards
    H_PER_SHARD = H // num_shards

    return MXFP8MOEBwdConfig(
        phase1_config=replace(tuned_config.phase1_config, M=B, K=H, N=I_TP_PER_SHARD),
        phase2_config=replace(tuned_config.phase2_config, M=B, K=2 * I_TP, N=H_PER_SHARD),
        phase3_config=replace(
            tuned_config.phase3_config,
            M=2 * I_TP if single_expert_dense else I_TP,
            K=B,
            N=H_PER_SHARD,
        ),
        phase4_config=replace(tuned_config.phase4_config, M=H_PER_SHARD, K=B, N=I_TP),
        phase3_transpose_mode=tuned_config.phase3_transpose_mode,
        phase4_transpose_mode=tuned_config.phase4_transpose_mode,
    )


def _validate_explicit_test_config(config: MXFP8MOEBwdConfig) -> None:
    """Reject partial test configs before they can reach kernel defaults."""
    for phase_name, phase_config in (
        ("phase1", config.phase1_config),
        ("phase2", config.phase2_config),
        ("phase3", config.phase3_config),
        ("phase4", config.phase4_config),
    ):
        block_m = phase_config.TILES_IN_BLOCK_M
        block_n = phase_config.TILES_IN_BLOCK_N
        block_k = phase_config.TILES_IN_BLOCK_K
        load_m = phase_config.TILES_IN_LOAD_M
        load_n = phase_config.TILES_IN_LOAD_N
        tile_n = phase_config.tile_n
        if block_m is None or block_n is None or block_k is None or load_m is None or load_n is None or tile_n is None:
            raise ValueError(f"{phase_name} must have explicit positive block and load tile counts")
        if min(block_m, block_n, block_k, load_m, load_n, tile_n) < 1:
            raise ValueError(f"{phase_name} must have explicit positive block and load tile counts")
        if block_m % load_m:
            raise ValueError(f"{phase_name} TILES_IN_LOAD_M must divide TILES_IN_BLOCK_M")
        if block_n % load_n:
            raise ValueError(f"{phase_name} TILES_IN_LOAD_N must divide TILES_IN_BLOCK_N")
        if block_m > 1 and load_m == 1:
            raise ValueError(f"{phase_name} multi-tile M block must not use a one-tile load")
        if block_n > 1 and load_n == 1:
            raise ValueError(f"{phase_name} multi-tile N block must not use a one-tile load")
        if load_n * tile_n > MAX_DGT_LOAD_WIDTH:
            raise ValueError(f"{phase_name} N load width must not exceed {MAX_DGT_LOAD_WIDTH}, got {load_n} * {tile_n}")


# The three-metric MXFP8 comparator is shared with the forward suite.
_moe_bwd_comparator = make_moe_validated_comparator


# ============================================================================
# Test parameter grid  (PARAM_NAMES + ABBREVS are imported from the shared module)
# ============================================================================

# fmt: off
# Per-method fast keys: only the (config, method) pairs that add unique
# branch coverage in the kernel source. Reuses the same shape as TEST_PARAMS
# (H, T, E, B, TOPK, I_TP).
_FAST_KEYS_BWD_VALIDATED: set[tuple] = {
    (1024, 1024, 4, 512, 4, 256),
}

# ============================================================================
# Selective xfail: tests that run but are expected to fail.
# Key: (H, T, E, B, TOPK, I_TP)  Value: reason string
# ============================================================================
XFAIL_PARAMS: dict[tuple, str] = {
}

# ============================================================================
# Selective skip: tests that are NOT run at all.
# Key: (H, T, E, B, TOPK, I_TP)  Value: reason string
# ============================================================================
SKIP_PARAMS: dict[tuple, str] = {
    (5120, 8192, 16, 4096, 4, 1024): "Static block expansion makes this shape prohibitively slow to compile.",
    (5120, 8192, 128, 4096, 1, 128): "Static block expansion makes this shape prohibitively slow to compile.",
    (6144, 4096, 16, 2048, 4, 1024): "Static block expansion makes this shape prohibitively slow to compile.",
    (6144, 4096, 16, 2048, 4, 128): "Static block expansion makes this shape prohibitively slow to compile.",
}

# ============================================================================
# Blocking sweep test params
# Each entry: [H, T, E, B, TOPK, I_TP, tiles_m, tiles_n, tiles_k]
# Uses pairwise covering array for (tiles_m, tiles_n, tiles_k) in {2, 4, 8}
# crossed with diverse shapes spanning small/large B, H, I_TP, and E values.
# _compute_blocking_params clamps values per-phase to not exceed available tiles.
# ============================================================================

# Shapes with B>=256 where blocking > 1 produces real multi-tile blocks.
# Each (shape, combo) entry below produces a unique effective blocking after
# clamping — no redundant tests. H={1024,2048,4096}, I_TP={384,768,1024}.
# Dims > 1: P1(K) from H, P2(N) from H/2, P2(K) from 2*I_TP,
# P3(M) from I_TP, P3(N) from H/2, P4(M) from H/2, P4(N) from I_TP.

# fmt: off
BLOCKING_TEST_PARAMS = [
    # H,    T,    E,  B,    TOPK, I_TP,  tm, tn, tk
    # --- B=512 entries ---
    # H=1024, I_TP=1024: P1(M=1,N=1,K=2) P2(M=1,N=1,K=4) P3(M=2,N=1,K=1) P4(M=1,N=2,K=1)
    [1024, 1024, 4,  512,  4,    1024,   2,  2,  2],  # P1(1,1,2) P2(1,1,2) P3(2,1,1) P4(1,2,1)
    # H=2048, I_TP=1024: P1(M=1,N=1,K=4) P2(M=1,N=2,K=4) P3(M=2,N=2,K=1) P4(M=2,N=2,K=1)
    [2048, 1024, 4,  512,  2,    1024,   2,  2,  2],  # P1(1,1,2) P2(1,2,2) P3(2,2,1) P4(2,2,1)
    # H=2048, I_TP=768: P1(M=1,N=1,K=4) P2(M=1,N=2,K=3) P3(M=1,N=2,K=1) P4(M=2,N=2,K=1)
    [2048, 1024, 2,  512,  1,    768,    2,  4,  4],  # P1(1,1,4) P2(1,2,3) P3(1,2,1) P4(2,2,1)
    # H=4096, I_TP=384: P1(M=1,N=1,K=8) P2(M=1,N=4,K=2) P3(M=1,N=4,K=1) P4(M=4,N=1,K=1)
    [4096, 1024, 4,  512,  2,    384,    2,  4,  4],  # P1(1,1,4) P2(1,4,2) P3(1,4,1) P4(4,1,1)
    [4096, 1024, 4,  512,  2,    384,    2,  8,  8],  # P1(1,1,8) P2(1,4,2) P3(1,4,1) P4(4,1,1)
    # --- B=256 entries ---
    # With B=256: P1/P2 M-tiles=ceil(256/128)=2, so //4=0→clamped to 1.
    # K-tiles for P3/P4=ceil(256/512)=1, always clamped.
    # Value of B=256 exercises smaller batch blocking in load/store paths.
    # H=2048, I_TP=1024: P1(K=4) P2(N=2,K=4) P3(N=2) P4(M=2,N=2)
    [2048, 1024, 4,  256,  2,    1024,   2,  2,  4],  # P1(1,1,4) P2(1,2,4) P3(2,2,1) P4(2,2,1)
    # H=4096, I_TP=1024: P1(K=8) P2(N=4,K=4) P3(N=4) P4(M=4,N=2)
    [4096, 1024, 4,  256,  2,    1024,   8,  8,  4],  # P1(1,1,4) P2(1,4,4) P3(2,4,1) P4(4,2,1)
    # H=4096, I_TP=384: P1(K=8) P2(N=4,K=2) P3(N=4) P4(M=4,N=1)
    [4096, 1024, 4,  256,  2,    384,    4,  4,  8],  # P1(1,1,8) P2(1,4,2) P3(1,4,1) P4(4,1,1)
    # H=2048, I_TP=768: P1(K=4) P2(N=2,K=3) P3(N=2) P4(M=2,N=2)
    [2048, 1024, 2,  256,  1,    768,    4,  4,  4],  # P1(1,1,4) P2(1,2,3) P3(1,2,1) P4(2,2,1)
]
# fmt: on

SPILL_RELOAD_BLOCKING_TEST_PARAMS = [
    BLOCKING_TEST_PARAMS[0],
    BLOCKING_TEST_PARAMS[4],
]

BLOCKING_CASES = [(*params, False) for params in BLOCKING_TEST_PARAMS] + [
    (*params, True) for params in SPILL_RELOAD_BLOCKING_TEST_PARAMS
]


# ============================================================================
# Large-T blocking test params
# Same [H, T, E, B, TOPK, I_TP, tiles_m, tiles_n, tiles_k] format as
# BLOCKING_TEST_PARAMS, isolated here because the large token count makes these
# substantially heavier than the standard blocking sweep.
# ============================================================================

# fmt: off
LARGE_T_BLOCKING_TEST_PARAMS = [
    # H,    T,      E,  B,    TOPK, I_TP,  tm, tn, tk
    # H=4096, I_TP=384, T=65536: max tiles 8 in M/N/K
    [4096, 65536, 2,  512,  2,    384,    8,  8,  8],  # P1(1,1,8) P2(1,4,2) P3(1,4,1) P4(4,1,1)
]
# fmt: on


def _build_params(params_list, fast_keys=None):
    """Build pytest params with this suite's fast/xfail/skip dicts applied."""
    return _build_params_common(params_list, fast_keys=fast_keys, skip_params=SKIP_PARAMS, xfail_params=XFAIL_PARAMS)


def _generate_sweep_params(shapes, num_cases, heavy_shapes):
    """Sample feature/blocking cases from the eligible sweep pool."""
    if not 10 <= num_cases <= 15:
        raise ValueError(f"Random sweep must collect 10-15 cases, got {num_cases}")

    excluded_shapes = set(SKIP_PARAMS)
    heavy_shapes = {tuple(shape) for shape in heavy_shapes}
    eligible_shapes = [tuple(shape) for shape in shapes if tuple(shape) not in excluded_shapes]
    unreasonable_shapes = [shape for shape in eligible_shapes if shape[3] < shape[1] // 4]
    if unreasonable_shapes:
        raise ValueError(f"Random sweep block size must be at least T/4, got {unreasonable_shapes}")

    normal_shapes = [shape for shape in eligible_shapes if shape not in heavy_shapes]
    eligible_heavy_shapes = [shape for shape in eligible_shapes if shape in heavy_shapes]

    heavy_case_count = min(1, len(eligible_heavy_shapes), num_cases)
    normal_case_count = num_cases - heavy_case_count
    if len(normal_shapes) < normal_case_count:
        raise ValueError(f"Random sweep requires {normal_case_count} normal shapes, found {len(normal_shapes)}")

    selected_shapes = random.sample(normal_shapes, normal_case_count)
    if heavy_case_count:
        selected_shapes.extend(random.sample(eligible_heavy_shapes, heavy_case_count))
    random.shuffle(selected_shapes)

    params = []
    clamp_options = (
        ("none", None),
        (
            "nonlinear",
            ClampLimits(non_linear_clamp_upper_limit=1.0, non_linear_clamp_lower_limit=-1.0),
        ),
        (
            "linear",
            ClampLimits(linear_clamp_upper_limit=0.5, linear_clamp_lower_limit=-0.5),
        ),
        (
            "both",
            ClampLimits(
                non_linear_clamp_upper_limit=1.0,
                non_linear_clamp_lower_limit=-1.0,
                linear_clamp_upper_limit=0.5,
                linear_clamp_lower_limit=-0.5,
            ),
        ),
    )

    # TODO: Replace independent random choices with a coverage-directed feature
    # matrix. For N binary features and K tests, balance each feature across 0/1
    # while maximizing pairwise Hamming distance; for K=2, use the all-zero and
    # all-one vectors. Reject duplicate vectors and any generation where a
    # feature stays constant across all tests, then apply the same principle to
    # categorical options. This maximizes feature coverage for a fixed test budget.
    for H, T, E, B, top_k, I_TP in selected_shapes:
        use_scale_packing = random.choice([True, False])
        bias = random.choice([True, False])
        prequantize_weights = random.choice([True, False])
        clamp_id, clamp_limits = random.choice(clamp_options)
        tiles_m = random.randint(1, 8)
        tiles_n = random.randint(1, 8)
        tiles_k = random.randint(1, 8)

        moe_bwd_config = _compute_moe_bwd_config(
            H=H,
            B=B,
            I_TP=I_TP,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
            spill_reload=False,
            use_scale_packing=use_scale_packing,
        )

        test_id = (
            f"hid_{H}_tok_{T}_exp_{E}_bs_{B}_k_{top_k}_int_{I_TP}"
            f"_sr0_sp{int(use_scale_packing)}_b{int(bias)}_pq{int(prequantize_weights)}"
            f"_cl{clamp_id}_tm{tiles_m}_tn{tiles_n}_tk{tiles_k}"
        )
        params.append(
            pytest.param(
                H,
                T,
                E,
                B,
                top_k,
                I_TP,
                {
                    "spill_reload": False,
                    "use_scale_packing": use_scale_packing,
                    "bias": bias,
                    "prequantize_weights": prequantize_weights,
                    "clamp_limits": clamp_limits,
                    "moe_bwd_config": moe_bwd_config,
                },
                id=test_id,
            )
        )

    return params


# ============================================================================
# Test class
# ============================================================================


@pytest_test_metadata(name="MoE MXFP8 Blockwise MatMul BWD Validated")
@pytest_marks(["moe_mxfp8", "moe", "blockwise_mm_bwd", "mxfp8", "validated"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMoeMxfp8BlockwiseMatMulBwdValidated:
    """Integration tests for MXFP8 MoE backward pass with three-metric validation."""

    def _run_test(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        run_with_lnc2: bool = True,
        moe_bwd_config=None,
        spill_reload: bool = False,
        use_scale_packing: bool = False,
        bias: bool = False,
        clamp_limits=None,
        prequantize_weights: bool = False,
        output_grad_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        down_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        d_gate_up_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        gate_up_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        d_gate_up_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        hidden_states_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        output_grad_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        scaled_intermediate_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
        phase3_transpose_mode: TransposeMode = TransposeMode.NC,
        phase4_transpose_mode: TransposeMode = TransposeMode.NC,
        single_expert_dense: bool = False,
        fast_dma_transpose: bool = False,
    ):
        T = tokens
        H = hidden
        I_TP = intermediate
        E = expert

        # Per-phase blocking resolution: explicit caller config -> exact shape-tuned table entry
        # -> explicit conservative test config. Tests never rely on kernel tile defaults.
        if moe_bwd_config is None:
            tuned_config = get_shape_tuned_config(
                B=block_size,
                H=H,
                I_TP=I_TP,
                num_shards=(2 if run_with_lnc2 else 1),
                shard_option=ShardOption.SHARD_ON_FREE,
                affinity_option=AffinityOption.AFFINITY_ON_I,
                compute_dtype=nl.bfloat16,
                spill_reload=spill_reload,
                use_scale_packing=use_scale_packing,
                bias=bias,
                single_expert_dense=single_expert_dense,
            )
            if tuned_config is not None:
                moe_bwd_config = _moe_bwd_config_from_tuned(
                    tuned_config,
                    H=H,
                    B=block_size,
                    I_TP=I_TP,
                    run_with_lnc2=run_with_lnc2,
                    single_expert_dense=single_expert_dense,
                )
            else:
                moe_bwd_config = _compute_moe_bwd_config(
                    H=H,
                    B=block_size,
                    I_TP=I_TP,
                    tiles_m=8,
                    tiles_n=4,
                    tiles_k=2,
                    run_with_lnc2=run_with_lnc2,
                    spill_reload=spill_reload,
                    use_scale_packing=use_scale_packing,
                    single_expert_dense=single_expert_dense,
                )

        _validate_explicit_test_config(moe_bwd_config)

        build_kwargs = {
            "tokens": T,
            "hidden": H,
            "intermediate": I_TP,
            "expert": E,
            "block_size": block_size,
            "top_k": top_k,
            "run_with_lnc2": run_with_lnc2,
            "moe_bwd_config": moe_bwd_config,
            "spill_reload": spill_reload,
            "use_scale_packing": use_scale_packing,
            "prequantize_weights": prequantize_weights,
            "bias": bias,
            "clamp_limits": clamp_limits,
            "single_expert_dense": single_expert_dense,
            "fast_dma_transpose": fast_dma_transpose,
        }
        transpose_modes = {
            "phase3_transpose_mode": phase3_transpose_mode,
            "phase4_transpose_mode": phase4_transpose_mode,
        }
        swizzle_modes = {
            "output_grad_swizzle_mode": output_grad_swizzle_mode,
            "down_weight_swizzle_mode": down_weight_swizzle_mode,
            "d_gate_up_swizzle_mode": d_gate_up_swizzle_mode,
            "gate_up_weight_swizzle_mode": gate_up_weight_swizzle_mode,
            "d_gate_up_t_swizzle_mode": d_gate_up_t_swizzle_mode,
            "hidden_states_t_swizzle_mode": hidden_states_t_swizzle_mode,
            "output_grad_t_swizzle_mode": output_grad_t_swizzle_mode,
            "scaled_intermediate_t_swizzle_mode": scaled_intermediate_t_swizzle_mode,
        }

        if prequantize_weights:
            kernel_inputs, orig_gate_up_weight, orig_down_weight = build_mxfp8_moe_bwd_inputs(**build_kwargs)
            kernel_inputs.update(transpose_modes)
            kernel_inputs.update(swizzle_modes)

            @functools.wraps(blockwise_mm_bwd_mxfp8_torch_ref)
            def _pq_torch_ref(**kwargs):
                kwargs["gate_up_proj_weight"] = torch.from_numpy(orig_gate_up_weight.astype(np.float32))
                kwargs["down_proj_weight"] = torch.from_numpy(orig_down_weight.astype(np.float32))
                return blockwise_mm_bwd_mxfp8_torch_ref(**kwargs)

            def input_gen(_):
                return kernel_inputs

            torch_ref = torch_ref_wrapper(_pq_torch_ref)
        else:

            def input_gen(_):
                kernel_inputs = build_mxfp8_moe_bwd_inputs(**build_kwargs)
                kernel_inputs.update(transpose_modes)
                kernel_inputs.update(swizzle_modes)
                return kernel_inputs

            torch_ref = torch_ref_wrapper(blockwise_mm_bwd_mxfp8_torch_ref)

        output_shapes = {
            "hidden_states_grad": ((T, H), bfloat16),
            "expert_affinities_masked_grad": ((T * E, 1), bfloat16),
            "gate_up_proj_weight_grad": ((E, H, 2, I_TP), bfloat16),
            "down_proj_weight_grad": ((E, I_TP, H), bfloat16),
        }
        if bias:
            output_shapes["gate_and_up_proj_bias_grad"] = ((E, 2, I_TP), bfloat16)
            output_shapes["down_proj_bias_grad"] = ((E, H), bfloat16)

        def output_tensors(kernel_input):
            return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in output_shapes.items()}

        lnc_count = 2 if run_with_lnc2 else 1
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd_mxfp8,
            torch_ref=torch_ref,
            kernel_input_generator=input_gen,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=common_dataclasses.CompilerArgs(
                logical_nc_config=lnc_count,
                platform_target=platform_target,
            ),
            custom_comparator=_moe_bwd_comparator(output_shapes),
        )

    # ------------------------------------------------------------------
    # Test: baseline shape coverage
    # ------------------------------------------------------------------

    # fmt: off
    _BASELINE_SHAPES = [
        # H,     T,     E,  B,    TOPK, I_TP
        # E=1, TopK=1
        [128,   1024,  1,  128,  1,    128],
        [384,   512,  1,  256,  1,    256],
        [1024,  1024,  1,  512,  1,    384],
        [2048,  1024,  1,  128,  1,    512],
        [384,   1024,  1,  256,  1,    512],
        # E=2, TopK=1
        [128,   1024,  2,  128,  1,    256],
        [1024,  1024,  2,  512,  1,    512],
        [2048,  1024,  2,  256,  1,    128],
        # E=2, TopK=2
        [384,   1024,  2,  512,  2,    384],
        [2048,  1024,  2,  128,  2,    256],
        # E=4, TopK=1
        [128,   1024,  4,  256,  1,    384],
        [1024,  1024,  4,  512,  1,    128],
        # E=4, TopK=2
        [384,   1024,  4,  128,  2,    512],
        [2048,  1024,  4,  256,  2,    384],
        # E=4, TopK=4
        [1024,  1024,  4,  512,  4,    256],
    ]
    # fmt: on

    @pytest_parametrize(PARAM_NAMES, _build_params(_BASELINE_SHAPES, _FAST_KEYS_BWD_VALIDATED), abbrevs=_ABBREVS)
    def test_moe_mxfp8_bwd_validated_baseline_shapes(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
        )

    _FEATURE_TEST_PARAMS = [
        pytest.param(512, 512, 2, 256, 2, 256, {"use_scale_packing": True}, id="scale_packing"),
        pytest.param(512, 1024, 1, 128, 1, 640, {"spill_reload": True}, id="spill_reload"),
        pytest.param(512, 512, 2, 256, 2, 256, {"bias": True}, id="bias"),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {
                "clamp_limits": ClampLimits(
                    non_linear_clamp_upper_limit=1.0,
                    non_linear_clamp_lower_limit=-1.0,
                )
            },
            id="clamp_nonlinear",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {
                "clamp_limits": ClampLimits(
                    linear_clamp_upper_limit=0.5,
                    linear_clamp_lower_limit=-0.5,
                )
            },
            id="clamp_linear",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"prequantize_weights": True, "use_scale_packing": True, "spill_reload": True},
            id="prequantized",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"phase3_transpose_mode": TransposeMode.DMA},
            id="phase3_dma",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"phase4_transpose_mode": TransposeMode.DMA},
            id="phase4_dma",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"d_gate_up_swizzle_mode": SwizzleMode.PE},
            id="d_gate_up_pe_swizzle_wrapx",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"d_gate_up_t_swizzle_mode": SwizzleMode.PE},
            id="d_gate_up_t_pe_swizzle_wrapx",
        ),
        pytest.param(
            512,
            512,
            2,
            256,
            2,
            256,
            {"scaled_intermediate_t_swizzle_mode": SwizzleMode.PE},
            id="scaled_intermediate_t_pe_swizzle_wrapx",
        ),
        pytest.param(
            512,
            512,
            1,
            512,
            1,
            256,
            {"single_expert_dense": True},
            id="single_expert_dense",
        ),
        pytest.param(512, 128, 1, 256, 1, 256, {}, id="block_larger_than_tokens"),
    ]

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _FEATURE_TEST_PARAMS,
    )
    def test_moe_mxfp8_bwd_validated_feature_paths(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        feature_kwargs: dict,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            **feature_kwargs,
        )

    # -----------------------------------------------------------------------------------
    # Test: blocking params sweep (TILES_IN_BLOCK_M/N/K from 1 to 8), spill_reload on/off
    # -----------------------------------------------------------------------------------

    @pytest_parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, tiles_m, tiles_n, tiles_k, spill_reload",
        BLOCKING_CASES,
        abbrevs={**_ABBREVS, "tiles_m": "tm", "tiles_n": "tn", "tiles_k": "tk", "spill_reload": "sr"},
    )
    def test_moe_mxfp8_bwd_validated_blocking(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        tiles_m: int,
        tiles_n: int,
        tiles_k: int,
        spill_reload: bool,
    ):
        """Test with non-default blocking params (TILES_IN_BLOCK_M/N/K > 1), spill_reload on/off."""
        moe_bwd_config = _compute_moe_bwd_config(
            H=hidden,
            B=block_size,
            I_TP=intermediate,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
            spill_reload=spill_reload,
            use_scale_packing=True,
        )
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            moe_bwd_config=moe_bwd_config,
            spill_reload=spill_reload,
            use_scale_packing=True,
        )

    # -----------------------------------------------------------------------------------
    # Test: large-T blocking params (max tiles 8 in M/N/K), spill_reload on/off
    # -----------------------------------------------------------------------------------

    @pytest.mark.skip(reason="E2E real-model use case (T=65536); too heavy for the standard suite, run manually.")
    @pytest_parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, tiles_m, tiles_n, tiles_k",
        LARGE_T_BLOCKING_TEST_PARAMS,
        abbrevs={**_ABBREVS, "tiles_m": "tm", "tiles_n": "tn", "tiles_k": "tk"},
    )
    def test_moe_mxfp8_bwd_validated_blocking_large_t(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        tiles_m: int,
        tiles_n: int,
        tiles_k: int,
    ):
        """Test large-T blocking params (TILES_IN_BLOCK_M/N/K up to 8), spill_reload on/off."""
        moe_bwd_config = _compute_moe_bwd_config(
            H=hidden,
            B=block_size,
            I_TP=intermediate,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
            spill_reload=True,
            use_scale_packing=True,
        )
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            moe_bwd_config=moe_bwd_config,
            spill_reload=True,
            use_scale_packing=True,
        )

    # ------------------------------------------------------------------
    # Test: Random sweep over all features and blocking params
    # ------------------------------------------------------------------

    # fmt: off
    _SWEEP_SHAPES = [
        # Large-token sweeps use B>=2048 to bound static block expansion.
        # B=128/256/512 behavior is covered by the T<=1024 test matrices above.
        # H,    T,    E,  B,    TOPK, I_TP
        [4096, 4096,  2, 2048, 2, 128],
        [4096, 4096,  4, 2048, 4, 128],
        [4096, 4096,  2, 2048, 2, 256],
        [4096, 4096,  4, 2048, 4, 256],
        [4096, 4096,  2, 2048, 2, 384],
        [4096, 4096,  4, 2048, 4, 384],
        [4096, 4096,  2, 2048, 2, 640],
        [4096, 4096,  4, 2048, 4, 640],
        [4096, 4096,  2, 2048, 2, 768],
        [4096, 4096,  4, 2048, 4, 768],
        [4096, 4096,  2, 2048, 2, 1024],
        [4096, 4096,  4, 2048, 4, 1024],
        [5120, 8192, 16, 4096, 1, 256],
        [5120, 8192, 16, 4096, 4, 1024],
        [5120, 8192, 128, 4096, 1, 128],
        [6144, 4096, 16, 2048, 4, 1024],
        [6144, 4096, 16, 2048, 4, 128],
        [6144, 4096, 1,  2048, 1, 128],
        # Affinity I test cases (from BF16 MoE BWD test_bwmm_bwd.py)
        [4096, 4096,  4, 2048, 2, 384],
        [4096, 4096,  4, 2048, 2, 1536],
        [5120, 4096,  4, 2048, 1, 2048],
        [2048, 4096,  2, 2048, 2, 768],

    ]
    # fmt: on

    _HEAVY_SWEEP_SHAPES = {
        (5120, 8192, 16, 4096, 1, 256),
        (6144, 4096, 1, 2048, 1, 128),
        (5120, 4096, 4, 2048, 1, 2048),
    }
    _SWEEP_CASES_PER_RUN = 12

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, sweep_kwargs",
        _generate_sweep_params(
            _SWEEP_SHAPES,
            _SWEEP_CASES_PER_RUN,
            _HEAVY_SWEEP_SHAPES,
        ),
    )
    def test_moe_mxfp8_bwd_validated_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        sweep_kwargs: dict,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            **sweep_kwargs,
        )

    # Required Qwen3-235B TP1/TP2 shapes at T=4096 and T=8192.
    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate",
        [
            pytest.param(4096, 4096, 1, 4096, 1, 1536, id="tp1_tok4096_bs4096"),
            pytest.param(4096, 4096, 1, 4096, 1, 768, id="tp2_tok4096_bs4096"),
            pytest.param(4096, 8192, 1, 2048, 1, 1536, id="tp1_tok8192_bs2048"),
            pytest.param(4096, 8192, 1, 4096, 1, 1536, id="tp1_tok8192_bs4096"),
            pytest.param(4096, 8192, 1, 2048, 1, 768, id="tp2_tok8192_bs2048"),
            pytest.param(4096, 8192, 1, 4096, 1, 768, id="tp2_tok8192_bs4096"),
        ],
    )
    def test_moe_mxfp8_bwd_validated_tuned_shapes(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
        )

    @pytest.mark.parametrize(
        "intermediate",
        [
            pytest.param(1536, id="tp1"),
            pytest.param(768, id="tp2"),
        ],
    )
    @pytest.mark.parametrize(
        "fast_dma_transpose",
        [
            pytest.param(False, id="fast_dma_off"),
            pytest.param(True, id="fast_dma_on"),
        ],
    )
    def test_moe_mxfp8_bwd_single_expert_dense_qwen3_235b(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        intermediate: int,
        fast_dma_transpose: bool,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=4096,
            tokens=4096,
            expert=1,
            block_size=4096,
            top_k=1,
            intermediate=intermediate,
            single_expert_dense=True,
            fast_dma_transpose=fast_dma_transpose,
            d_gate_up_t_swizzle_mode=SwizzleMode.PE,
            hidden_states_t_swizzle_mode=SwizzleMode.PE,
            output_grad_t_swizzle_mode=SwizzleMode.PE,
            scaled_intermediate_t_swizzle_mode=SwizzleMode.PE,
        )

    @pytest.mark.parametrize(
        "intermediate",
        [
            pytest.param(1536, id="tp1"),
        ],
    )
    @pytest.mark.parametrize(
        "fast_dma_transpose",
        [
            pytest.param(False, id="fast_dma_off"),
            pytest.param(True, id="fast_dma_on"),
        ],
    )
    def test_moe_mxfp8_bwd_qwen3_235b_8k_two_blocks_single_expert_dense(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        intermediate: int,
        fast_dma_transpose: bool,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=4096,
            tokens=8192,
            expert=1,
            block_size=4096,
            top_k=1,
            intermediate=intermediate,
            single_expert_dense=True,
            fast_dma_transpose=fast_dma_transpose,
            d_gate_up_t_swizzle_mode=SwizzleMode.PE,
            hidden_states_t_swizzle_mode=SwizzleMode.PE,
            output_grad_t_swizzle_mode=SwizzleMode.PE,
            scaled_intermediate_t_swizzle_mode=SwizzleMode.PE,
        )
