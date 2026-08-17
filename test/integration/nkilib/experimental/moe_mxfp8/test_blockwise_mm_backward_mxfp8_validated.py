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
from dataclasses import dataclass
from typing import final

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import (
    L_TILE_K,
    TILE_M,
    TILE_N,
)
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import ClampLimits
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import blockwise_mm_bwd_mxfp8
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8_torch import (
    blockwise_mm_bwd_mxfp8_torch_ref,
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
from test.utils import common_dataclasses
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16

# ============================================================================
# Blocking params helper (backward-specific: 4 phases)
# ============================================================================


@dataclass
class BlockingParams:
    """Container for per-phase MatmulMxfp8KernelConfig blocking overrides."""

    phase1: MatmulMxfp8KernelConfig
    phase2: MatmulMxfp8KernelConfig
    phase3: MatmulMxfp8KernelConfig
    phase4: MatmulMxfp8KernelConfig


def _compute_blocking_params(
    H: int,
    B: int,
    I_TP: int,
    tiles_m: int,
    tiles_n: int,
    tiles_k: int,
    run_with_lnc2: bool = True,
) -> BlockingParams:
    """Compute valid BlockingParams for the given shape dimensions.

    The kernel internally multiplies TILES_IN_BLOCK_M by 4, so the effective
    M-blocking is 4x what we pass. We clamp to available tiles // 4 to avoid
    exceeding the actual dimension.

    Args:
        tiles_m: Desired TILES_IN_BLOCK_M (1-8 range).
        tiles_n: Desired TILES_IN_BLOCK_N (1-8 range).
        tiles_k: Desired TILES_IN_BLOCK_K (1-8 range).
    """
    num_shards = 2 if run_with_lnc2 else 1
    I_TP_PER_SHARD = I_TP // num_shards
    H_PER_SHARD = H // num_shards

    # Phase 1: output_grad[B, H] @ W_down[H, I_TP/shard]
    # M maps to B (tile_m=128), N to I_TP_PER_SHARD (tile_n=512), K to H (l_tile_k=512)
    p1_num_b_tiles = math.ceil(B / TILE_M)
    p1_num_i_tiles = math.ceil(I_TP_PER_SHARD / TILE_N)
    p1_num_k_tiles = math.ceil(H / L_TILE_K)
    phase1 = MatmulMxfp8KernelConfig(
        M=B,
        K=H,
        N=I_TP_PER_SHARD,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, p1_num_b_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, p1_num_i_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p1_num_k_tiles),
    )

    # Phase 2: d_gate_up[B, 2*I_TP] @ W_gate_up[2*I_TP, H/shard]
    # M maps to B (tile_m=128), N to H_PER_SHARD (tile_n=512), K to 2*I_TP (l_tile_k=512)
    p2_num_b_tiles = math.ceil(B / TILE_M)
    p2_num_h_tiles = math.ceil(H_PER_SHARD / TILE_N)
    p2_num_k_tiles = math.ceil((2 * I_TP) / L_TILE_K)
    phase2 = MatmulMxfp8KernelConfig(
        M=B,
        K=2 * I_TP,
        N=H_PER_SHARD,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, p2_num_b_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, p2_num_h_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p2_num_k_tiles),
    )

    # Phase 3: d_gate_up_T[I_TP, B] @ hidden_states_T[B, H/shard]
    # M maps to I_TP (tile_m=128), N to H_PER_SHARD (tile_n=512), K to B (l_tile_k=512)
    p3_num_i_tiles = math.ceil(I_TP / TILE_M)
    p3_num_h_tiles = math.ceil(H_PER_SHARD / TILE_N)
    p3_num_k_tiles = math.ceil(B / L_TILE_K)
    phase3 = MatmulMxfp8KernelConfig(
        M=I_TP,
        K=B,
        N=H_PER_SHARD,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, p3_num_i_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, p3_num_h_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p3_num_k_tiles),
    )

    # Phase 4: output_grad_T[H/shard, B] @ scaled_intermediate_T[B, I_TP]
    # M maps to H_PER_SHARD (tile_m=128), N to I_TP (tile_n=512), K to B (l_tile_k=512)
    p4_num_h_tiles = math.ceil(H_PER_SHARD / TILE_M)
    p4_num_i_tiles = math.ceil(I_TP / TILE_N)
    p4_num_k_tiles = math.ceil(B / L_TILE_K)
    phase4 = MatmulMxfp8KernelConfig(
        M=H_PER_SHARD,
        K=B,
        N=I_TP,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, p4_num_h_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, p4_num_i_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, p4_num_k_tiles),
    )

    return BlockingParams(phase1=phase1, phase2=phase2, phase3=phase3, phase4=phase4)


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
_FAST_KEYS_BWD_VALIDATED_SPILL_RELOAD: set[tuple] = {
    (512, 1024, 1, 128, 1, 640),
}
_FAST_KEYS_BWD_VALIDATED_LARGE_T: set[tuple] = set()  # No fast configs needed.
_FAST_KEYS_BWD_VALIDATED_BIAS: set[tuple] = {
    (1024, 1024, 2, 256, 2, 256),
}
_FAST_KEYS_BWD_PREQUANTIZED: set[tuple] = set()  # No fast configs needed.
_FAST_KEYS_BWD_PREQUANTIZED_NO_SCALE_PACKING: set[tuple] = {
    (512, 1024, 1, 128, 1, 640),
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
    [2048, 1024, 4,  256,  2,    1024,   4,  4,  8],  # P1(1,1,4) P2(1,2,4) P3(2,2,1) P4(2,2,1)
    # H=4096, I_TP=1024: P1(K=8) P2(N=4,K=4) P3(N=4) P4(M=4,N=2)
    [4096, 1024, 4,  256,  2,    1024,   8,  8,  4],  # P1(1,1,4) P2(1,4,4) P3(2,4,1) P4(4,2,1)
    # H=4096, I_TP=384: P1(K=8) P2(N=4,K=2) P3(N=4) P4(M=4,N=1)
    [4096, 1024, 4,  256,  2,    384,    4,  4,  8],  # P1(1,1,8) P2(1,4,2) P3(1,4,1) P4(4,1,1)
    # H=2048, I_TP=768: P1(K=4) P2(N=2,K=3) P3(N=2) P4(M=2,N=2)
    [2048, 1024, 2,  256,  1,    768,    4,  4,  4],  # P1(1,1,4) P2(1,2,3) P3(1,2,1) P4(2,2,1)
]
# fmt: on


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


def _generate_sweep_params(shapes, num_configs_per_shape, seed=42):
    """Generate randomized test params: each shape x num_configs random feature/blocking combos."""
    import random

    params = []
    for shape in shapes:
        H, T, E, B, top_k, I_TP = shape
        rng = random.Random(seed ^ hash(tuple(shape)))

        for _i in range(num_configs_per_shape):
            spill_reload = rng.choice([True, False])
            use_scale_packing = rng.choice([True, False])
            bias = rng.choice([True, False])
            prequantize_weights = rng.choice([True, False])
            clamp_limits = rng.choice(
                [
                    None,
                    ClampLimits(non_linear_clamp_upper_limit=1.0, non_linear_clamp_lower_limit=-1.0),
                    ClampLimits(linear_clamp_upper_limit=0.5, linear_clamp_lower_limit=-0.5),
                    ClampLimits(
                        non_linear_clamp_upper_limit=1.0,
                        non_linear_clamp_lower_limit=-1.0,
                        linear_clamp_upper_limit=0.5,
                        linear_clamp_lower_limit=-0.5,
                    ),
                ]
            )

            tiles_m = rng.randint(1, 8)
            tiles_n = rng.randint(1, 8)
            tiles_k = rng.randint(1, 8)

            blocking = _compute_blocking_params(
                H=H,
                B=B,
                I_TP=I_TP,
                tiles_m=tiles_m,
                tiles_n=tiles_n,
                tiles_k=tiles_k,
            )

            test_id = (
                f"hid_{H}_tok_{T}_exp_{E}_bs_{B}_k_{top_k}_int_{I_TP}"
                f"_sr{int(spill_reload)}_sp{int(use_scale_packing)}"
                f"_b{int(bias)}_pq{int(prequantize_weights)}"
                f"_tm{tiles_m}_tn{tiles_n}_tk{tiles_k}"
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
                        "spill_reload": spill_reload,
                        "use_scale_packing": use_scale_packing,
                        "bias": bias,
                        "prequantize_weights": prequantize_weights,
                        "clamp_limits": clamp_limits,
                        "blocking_params": blocking,
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
        blocking_params=None,
        spill_reload: bool = False,
        use_scale_packing: bool = False,
        bias: bool = False,
        clamp_limits=None,
        prequantize_weights: bool = False,
    ):
        T = tokens
        H = hidden
        I_TP = intermediate
        E = expert

        build_kwargs = {
            "tokens": T,
            "hidden": H,
            "intermediate": I_TP,
            "expert": E,
            "block_size": block_size,
            "top_k": top_k,
            "run_with_lnc2": run_with_lnc2,
            "blocking_params": blocking_params,
            "spill_reload": spill_reload,
            "use_scale_packing": use_scale_packing,
            "prequantize_weights": prequantize_weights,
            "bias": bias,
            "clamp_limits": clamp_limits,
        }

        if prequantize_weights:
            kernel_inputs, orig_gate_up_weight, orig_down_weight = build_mxfp8_moe_bwd_inputs(**build_kwargs)

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
                return build_mxfp8_moe_bwd_inputs(**build_kwargs)

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
                additional_cmd_args=[
                    "--internal-backend-options=--skip-pass=address_rotation_sb --skip-pass=address_rotation_psum"
                ],
            ),
            custom_comparator=_moe_bwd_comparator(output_shapes),
        )

    # ------------------------------------------------------------------
    # Test: All features False (spill_reload=False, scale_packing=False)
    # ------------------------------------------------------------------

    # fmt: off
    _ALL_FEATURES= [
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

    _FEATURE_CONFIGS = [
        pytest.param({}, id="baseline"),
        pytest.param({"use_scale_packing": True}, id="scale_packing"),
        pytest.param({"spill_reload": True}, id="spill_reload"),
        pytest.param({"bias": True}, id="bias"),
        pytest.param(
            {
                "clamp_limits": ClampLimits(
                    non_linear_clamp_upper_limit=1.0,
                    non_linear_clamp_lower_limit=-1.0,
                    linear_clamp_upper_limit=0.5,
                    linear_clamp_lower_limit=-0.5,
                )
            },
            id="clamp",
        ),
        pytest.param({"prequantize_weights": True, "use_scale_packing": True, "spill_reload": True}, id="prequantized"),
    ]

    @pytest_parametrize(PARAM_NAMES, _build_params(_ALL_FEATURES, _FAST_KEYS_BWD_VALIDATED), abbrevs=_ABBREVS)
    @pytest.mark.parametrize("feature_kwargs", _FEATURE_CONFIGS)
    def test_moe_mxfp8_bwd_validated(
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
        defaults = {"spill_reload": False, "use_scale_packing": False}
        defaults.update(feature_kwargs)
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            **defaults,
        )

    # -----------------------------------------------------------------------------------
    # Test: blocking params sweep (TILES_IN_BLOCK_M/N/K from 1 to 8), spill_reload on/off
    # -----------------------------------------------------------------------------------

    @pytest_parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, tiles_m, tiles_n, tiles_k",
        BLOCKING_TEST_PARAMS,
        abbrevs={**_ABBREVS, "tiles_m": "tm", "tiles_n": "tn", "tiles_k": "tk"},
    )
    @pytest.mark.parametrize(
        "spill_reload",
        [pytest.param(True, id="sr1"), pytest.param(False, id="sr0")],
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
        blocking = _compute_blocking_params(
            H=hidden,
            B=block_size,
            I_TP=intermediate,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
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
            blocking_params=blocking,
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
        blocking = _compute_blocking_params(
            H=hidden,
            B=block_size,
            I_TP=intermediate,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
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
            blocking_params=blocking,
            spill_reload=True,
            use_scale_packing=True,
        )

    # ------------------------------------------------------------------
    # Test: Random sweep over all features and blocking params
    # ------------------------------------------------------------------

    # fmt: off
    _SWEEP_SHAPES = [
        # H,    T,    E,  B,    TOPK, I_TP
        [4096, 4096,  2, 128,  2, 128],
        [4096, 4096,  4, 128,  4, 128],
        [4096, 4096,  2, 128,  2, 256],
        [4096, 4096,  4, 128,  4, 256],
        [4096, 4096,  2, 128,  2, 384],
        [4096, 4096,  4, 128,  4, 384],
        [4096, 4096,  2, 128,  2, 640],
        [4096, 4096,  4, 128,  4, 640],
        [4096, 4096,  2, 128,  2, 768],
        [4096, 4096,  4, 128,  4, 768],
        [4096, 4096,  2, 128,  2, 1024],
        [4096, 4096,  4, 128,  4, 1024],
        [5120, 8192, 16,  512, 1, 256],
        [5120, 8192, 16,  256, 4, 1024],
        [5120, 8192, 128, 256, 1, 128],
        [6144, 4096, 16,  512, 4, 1024],
        [6144, 4096, 16,  512, 4, 128],
        [6144, 4096, 1,   512, 1, 128],
        [4096, 4096, 2,   512, 2, 384],
        [4096, 4096, 4,   512, 2, 384],
        # Affinity I test cases (from BF16 MoE BWD test_bwmm_bwd.py)
        [4096, 4096,  4, 128,  2, 384],
        [4096, 4096,  4, 256,  2, 384],
        [4096, 4096,  4, 128,  2, 1536],
        [4096, 4096,  4, 256,  2, 1536],
        [5120, 4096,  4, 128,  1, 2048],
        [5120, 4096,  4, 256,  1, 2048],
        [2048, 4096,  2, 128,  2, 768],

    ]
    # fmt: on

    _SWEEP_CONFIGS_PER_SHAPE = 3

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, sweep_kwargs",
        _generate_sweep_params(_SWEEP_SHAPES, _SWEEP_CONFIGS_PER_SHAPE),
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
