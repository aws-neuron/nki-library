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

"""Integration tests for MXFP8 MoE forward pass with check_correctness validation.

Mirrors the MXFP8 MoE backward validated suite. Uses the same three-metric custom
validator (cosine similarity, normalized Euclidean distance, allclose with scaled
atol) and validates every emitted output: the layer output and both activation
checkpoints (gate_up_proj_act_checkpoint_T, scaled_intermediate_checkpoint_T).
"""

import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace
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
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8 import (
    _validate_inputs_and_derive_dims,
    blockwise_mm_fwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    blockwise_mm_fwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig

from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import (
    build_mxfp8_moe_fwd_inputs,
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
# Blocking params helper (forward-specific: 2 phases, gate/up + down)
# ============================================================================


@dataclass
class BlockingParams:
    """Container for per-phase MatmulMxfp8KernelConfig blocking overrides.

    The forward has two matmul phases (gate/up and down), versus the backward's
    four.
    """

    gate_up: MatmulMxfp8KernelConfig
    down: MatmulMxfp8KernelConfig


def _compute_blocking_params(
    H: int,
    B: int,
    I_TP: int,
    tiles_m: int,
    tiles_n: int,
    tiles_k: int,
) -> BlockingParams:
    """Compute valid BlockingParams for the given shape dimensions.

    SHARD_ON_BLOCK partitions whole blocks across cores, so each block's two
    GEMMs run on a single core over the full (unsharded) H / I_TP — the blocking
    math does not divide by num_shards (unlike the backward's SHARD_ON_FREE).

    The M-block factor is effectively multiplied by 4 inside the matmul, so we
    clamp to available tiles // 4 to avoid exceeding the dimension.

    Args:
        tiles_m: Desired TILES_IN_BLOCK_M (1-8 range).
        tiles_n: Desired TILES_IN_BLOCK_N (1-8 range).
        tiles_k: Desired TILES_IN_BLOCK_K (1-8 range).
    """
    # Gate/up: hidden_block[B, H] @ W_gate_up[H, 2*I_TP]
    # M -> B (tile_m=128), N -> I_TP (tile_n=512), K -> H (l_tile_k=512)
    gu_num_b_tiles = math.ceil(B / TILE_M)
    gu_num_i_tiles = math.ceil(I_TP / TILE_N)
    gu_num_k_tiles = math.ceil(H / L_TILE_K)
    gate_up = MatmulMxfp8KernelConfig(
        M=B,
        K=H,
        N=I_TP,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, gu_num_b_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, gu_num_i_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, gu_num_k_tiles),
    )

    # Down: scaled_intermediate[B, I_TP] @ W_down[I_TP, H]
    # M -> B (tile_m=128), N -> H (tile_n=512), K -> I_TP (l_tile_k=512)
    d_num_b_tiles = math.ceil(B / TILE_M)
    d_num_h_tiles = math.ceil(H / TILE_N)
    d_num_k_tiles = math.ceil(I_TP / L_TILE_K)
    down = MatmulMxfp8KernelConfig(
        M=B,
        K=I_TP,
        N=H,
        TILES_IN_BLOCK_M=_clamp_tiles(tiles_m, max(1, d_num_b_tiles // 4)),
        TILES_IN_BLOCK_N=_clamp_tiles(tiles_n, d_num_h_tiles),
        TILES_IN_BLOCK_K=_clamp_tiles(tiles_k, d_num_k_tiles),
    )

    return BlockingParams(gate_up=gate_up, down=down)


# The three-metric MXFP8 comparator is shared with the backward suite.
_moe_fwd_comparator = make_moe_validated_comparator


# ============================================================================
# Test parameter grid  (PARAM_NAMES + ABBREVS are imported from the shared module)
# ============================================================================

# fmt: off
# Per-method fast keys: only the (config, method) pairs that add unique
# branch coverage. Reuses the same shape tuple as the param lists
# (H, T, E, B, TOPK, I_TP).
_FAST_KEYS_FWD_VALIDATED: set[tuple] = {
    (1024, 1024, 4, 512, 4, 256),
}

# ============================================================================
# Selective xfail: tests that run but are expected to fail.
# Key: (H, T, E, B, TOPK, I_TP)  Value: reason string
# ============================================================================
XFAIL_PARAMS: dict[tuple, str] = {
}

# Grad clamp (ClampLimits) configs are expected to fail under the current
# validation methodology. strict=False so an occasional numeric pass does not
# fail the suite.
GRAD_CLAMP_XFAIL_REASON = (
    "Current validation methodology over penalizes the quantization error amplified by grad clamping, "
    "causing the tests to fail. TODO: design a better method to test grad clamp"
)

# Single-block (N==1, i.e. block_size == tokens) configs run under LNC2 with
# SHARD_ON_BLOCK, which leaves one core with no block: half the output tiles are
# never written. These fail the kernel_assert(N >= num_shards) guard. A col-parallel
# single-block path (SHARD_ON_FREE over I_TP / H) is planned; xfail until it lands.
SINGLE_BLOCK_XFAIL_REASON = (
    "N==1 (block_size == tokens) under LNC2 SHARD_ON_BLOCK leaves one core idle; "
    "guarded by kernel_assert(N >= num_shards). TODO: implement col-parallel single-block path."
)

# ============================================================================
# Selective skip: tests that are NOT run at all.
# Key: (H, T, E, B, TOPK, I_TP)  Value: reason string
# ============================================================================
SKIP_PARAMS: dict[tuple, str] = {
}


# ============================================================================
# Blocking sweep test params
# Each entry: [H, T, E, B, TOPK, I_TP, tiles_m, tiles_n, tiles_k]
# Pairwise covering array for (tiles_m, tiles_n, tiles_k) in {2, 4, 8}
# crossed with diverse shapes. _compute_blocking_params clamps per-phase.
# ============================================================================

# fmt: off
BLOCKING_TEST_PARAMS = [
    # H,    T,    E,  B,    TOPK, I_TP,  tm, tn, tk
    # --- B=512 entries ---
    [1024, 1024, 4,  512,  4,    1024,   2,  2,  2],
    [2048, 1024, 4,  512,  2,    1024,   2,  2,  2],
    [2048, 1024, 2,  512,  1,    768,    2,  4,  4],
    [4096, 1024, 4,  512,  2,    384,    2,  4,  4],
    [4096, 1024, 4,  512,  2,    384,    2,  8,  8],
    # --- B=256 entries ---
    [2048, 1024, 4,  256,  2,    1024,   2,  2,  4],
    [2048, 1024, 4,  256,  2,    1024,   4,  4,  8],
    [4096, 1024, 4,  256,  2,    1024,   8,  8,  4],
    [4096, 1024, 4,  256,  2,    384,    4,  4,  8],
    [2048, 1024, 2,  256,  1,    768,    4,  4,  4],
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
            # Pre-quantized weights not yet supported in the forward (see kernel).
            prequantize_weights = False
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

            # Independently randomize which activation checkpoints are saved so the
            # sweep also exercises the skip paths (checkpoint not computed/stored/
            # returned) under random shapes, not just the default both-saved case.
            # Drawn last so the existing feature/tile draws (and thus test ids)
            # stay stable. All four flag combinations are valid forward configs.
            save_gate_up_proj_act = rng.choice([True, False])
            save_scaled_intermediate = rng.choice([True, False])
            checkpoint_config = MXFP8MOECheckpointConfig(
                save_gate_up_proj_act=save_gate_up_proj_act,
                save_scaled_intermediate=save_scaled_intermediate,
            )

            test_id = (
                f"hid_{H}_tok_{T}_exp_{E}_bs_{B}_k_{top_k}_int_{I_TP}"
                f"_sr{int(spill_reload)}_sp{int(use_scale_packing)}"
                f"_b{int(bias)}_pq{int(prequantize_weights)}"
                f"_tm{tiles_m}_tn{tiles_n}_tk{tiles_k}"
                f"_cg{int(save_gate_up_proj_act)}_cs{int(save_scaled_intermediate)}"
            )

            # xfail any generated config that enables it.
            marks = (
                [pytest.mark.xfail(reason=GRAD_CLAMP_XFAIL_REASON, strict=False)] if clamp_limits is not None else []
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
                        "checkpoint_config": checkpoint_config,
                    },
                    id=test_id,
                    marks=marks,
                )
            )
    return params


def _fake_tensor(shape, dtype=nl.bfloat16):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _build_no_indirect_validator_inputs(
    tokens=256,
    hidden=128,
    intermediate=128,
    expert=1,
    block_size=128,
    token_position_shape=(1,),
    block_to_expert_shape=(1, 1),
):
    return {
        "hidden_states": _fake_tensor((tokens, hidden)),
        "gate_up_proj_weight": _fake_tensor((expert, 2, intermediate, hidden)),
        "down_proj_weight": _fake_tensor((expert, hidden, intermediate)),
        "token_position_to_id": _fake_tensor(token_position_shape, dtype=nl.int32),
        "block_to_expert": _fake_tensor(block_to_expert_shape, dtype=nl.int32),
        "expert_affinities_masked": _fake_tensor((tokens * expert, 1)),
        "block_size": block_size,
        "num_shards": 2,
        "no_indirect_load": True,
    }


# ============================================================================
# no_indirect_load input contract tests
# ============================================================================


@pytest.mark.fast
def test_moe_mxfp8_fwd_no_indirect_load_builder_contract():
    inputs = build_mxfp8_moe_fwd_inputs(
        tokens=256,
        hidden=128,
        intermediate=128,
        expert=1,
        block_size=128,
        top_k=1,
        no_indirect_load=True,
    )

    assert inputs["no_indirect_load"]
    assert inputs["token_position_to_id"].shape == (1,)
    assert inputs["block_to_expert"].shape == (1, 1)
    assert inputs["gate_up_proj_weight"].shape[0] == 1
    assert inputs["down_proj_weight"].shape[0] == 1
    assert inputs["expert_affinities_masked"].shape == (256, 1)
    assert not inputs["is_tensor_update_accumulating"]
    assert not inputs["skip_dma"].skip_token
    assert not inputs["skip_dma"].skip_weight


@pytest.mark.fast
@pytest.mark.parametrize(
    "override_kwargs, match",
    [
        pytest.param({"expert": 2}, "expert=1", id="multiple_experts"),
        pytest.param({"top_k": 2}, "top_k=1", id="top_k_gt_one"),
        pytest.param({"tokens": 768, "block_size": 512}, "tokens divisible by block_size", id="partial_block"),
    ],
)
def test_moe_mxfp8_fwd_no_indirect_load_input_contract(override_kwargs, match):
    kwargs = {
        "tokens": 1024,
        "hidden": 128,
        "intermediate": 128,
        "expert": 1,
        "block_size": 128,
        "top_k": 1,
        "no_indirect_load": True,
    }
    kwargs.update(override_kwargs)
    with pytest.raises(AssertionError, match=match):
        build_mxfp8_moe_fwd_inputs(**kwargs)


@pytest.mark.fast
def test_moe_mxfp8_fwd_no_indirect_load_validator_derives_contiguous_blocks():
    assert _validate_inputs_and_derive_dims(**_build_no_indirect_validator_inputs(tokens=512, block_size=256)) == (
        512,
        128,
        128,
        1,
        2,
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "override_kwargs, match",
    [
        pytest.param(
            {
                "gate_up_proj_weight": _fake_tensor((2, 2, 128, 128)),
                "down_proj_weight": _fake_tensor((2, 128, 128)),
                "expert_affinities_masked": _fake_tensor((512, 1)),
            },
            "requires exactly one expert weight",
            id="multiple_expert_weights",
        ),
        pytest.param(
            {"token_position_to_id": _fake_tensor((256,), dtype=nl.int32)},
            "dummy token_position_to_id shape",
            id="non_dummy_token_position",
        ),
        pytest.param(
            {"block_to_expert": _fake_tensor((2, 1), dtype=nl.int32)},
            "dummy block_to_expert shape",
            id="non_dummy_block_to_expert",
        ),
        pytest.param(
            {
                "hidden_states": _fake_tensor((384, 128)),
                "expert_affinities_masked": _fake_tensor((384, 1)),
                "block_size": 256,
            },
            "T to be divisible by block_size",
            id="partial_block",
        ),
    ],
)
def test_moe_mxfp8_fwd_no_indirect_load_kernel_contract(override_kwargs, match):
    kwargs = _build_no_indirect_validator_inputs()
    kwargs.update(override_kwargs)
    with pytest.raises(AssertionError, match=match):
        _validate_inputs_and_derive_dims(**kwargs)


# ============================================================================
# Test class
# ============================================================================


@pytest_test_metadata(name="MoE MXFP8 Blockwise MatMul FWD Validated")
@pytest_marks(["moe_mxfp8", "moe", "blockwise_mm_fwd", "mxfp8", "validated"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMoeMxfp8BlockwiseMatMulFwdValidated:
    """Integration tests for MXFP8 MoE forward pass with three-metric validation."""

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
        checkpoint_config=None,
        no_indirect_load: bool = False,
    ):
        T = tokens
        H = hidden
        I_TP = intermediate
        E = expert

        if checkpoint_config is None:
            checkpoint_config = MXFP8MOECheckpointConfig()

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
            "checkpoint_config": checkpoint_config,
            "no_indirect_load": no_indirect_load,
        }

        if prequantize_weights:
            kernel_inputs, orig_gate_up_weight, orig_down_weight = build_mxfp8_moe_fwd_inputs(**build_kwargs)

            @functools.wraps(blockwise_mm_fwd_mxfp8_torch_ref)
            def _pq_torch_ref(**kwargs):
                kwargs["gate_up_proj_weight"] = torch.from_numpy(orig_gate_up_weight.astype(np.float32))
                kwargs["down_proj_weight"] = torch.from_numpy(orig_down_weight.astype(np.float32))
                return blockwise_mm_fwd_mxfp8_torch_ref(**kwargs)

            def input_gen(_):
                return kernel_inputs

            torch_ref = torch_ref_wrapper(_pq_torch_ref)
        else:

            def input_gen(_):
                return build_mxfp8_moe_fwd_inputs(**build_kwargs)

            torch_ref = torch_ref_wrapper(blockwise_mm_fwd_mxfp8_torch_ref)

        # Output names match the kernel's positional outputs (and the torch ref's
        # dict keys); each checkpoint is present only when its save flag is set.
        output_shapes = {
            "output_hidden_states": ((T, H), bfloat16),
        }
        if checkpoint_config.save_gate_up_proj_act:
            output_shapes["gate_up_proj_act_checkpoint_T"] = (
                (self._n_blocks(T, top_k, E, block_size), 2, I_TP, block_size),
                bfloat16,
            )
        if checkpoint_config.save_scaled_intermediate:
            output_shapes["scaled_intermediate_checkpoint_T"] = (
                (self._n_blocks(T, top_k, E, block_size), I_TP, block_size),
                bfloat16,
            )

        def output_tensors(kernel_input):
            return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in output_shapes.items()}

        lnc_count = 2 if run_with_lnc2 else 1
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_fwd_mxfp8,
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
            custom_comparator=_moe_fwd_comparator(output_shapes),
        )

    @staticmethod
    def _n_blocks(T, top_k, E, B):
        """Match the kernel's block count: N = ceil((T*top_k - (E-1)) / B) + (E-1)."""
        return math.ceil((T * top_k - (E - 1)) / B) + E - 1

    # ------------------------------------------------------------------
    # Test: all feature configs over a spread of shapes
    # ------------------------------------------------------------------

    # fmt: off
    _ALL_FEATURES = [
        # H,     T,     E,  B,    TOPK, I_TP
        # E=1, TopK=1
        [128,   1024,  1,  128,  1,    128],
        [384,   512,   1,  256,  1,    256],
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
            marks=pytest.mark.xfail(reason=GRAD_CLAMP_XFAIL_REASON, strict=False),
        ),
        # NOTE: pre-quantized weights are not yet supported by the forward kernel
        # (needs a forward-natural x4 layout); the "prequantized" config is omitted
        # until that path lands. See blockwise_mm_forward_mxfp8._validate_kernel_options.
        pytest.param(
            {"checkpoint_config": MXFP8MOECheckpointConfig(save_scaled_intermediate=False)},
            id="no_scaled_ckpt",
        ),
        pytest.param(
            {"checkpoint_config": MXFP8MOECheckpointConfig(save_gate_up_proj_act=False)},
            id="no_gate_up_ckpt",
        ),
    ]

    _NO_INDIRECT_LOAD_PARAMS = [
        pytest.param(128, 1024, 1, 128, 1, 128, {}, marks=pytest.mark.fast, id="b128_baseline"),
        pytest.param(
            384,
            512,
            1,
            256,
            1,
            256,
            {"use_scale_packing": True},
            marks=pytest.mark.fast,
            id="b256_scale_packing",
        ),
        pytest.param(
            1024,
            1024,
            1,
            512,
            1,
            384,
            {"spill_reload": True},
            marks=pytest.mark.fast,
            id="b512_spill_reload",
        ),
        pytest.param(
            2048,
            1024,
            1,
            128,
            1,
            512,
            {
                "clamp_limits": ClampLimits(
                    non_linear_clamp_upper_limit=1.0,
                    non_linear_clamp_lower_limit=-1.0,
                    linear_clamp_upper_limit=0.5,
                    linear_clamp_lower_limit=-0.5,
                ),
                "checkpoint_config": MXFP8MOECheckpointConfig(save_scaled_intermediate=False),
            },
            # clamp_limits set -> xfail like every other grad-clamp config (strict=False).
            marks=[pytest.mark.fast, pytest.mark.xfail(reason=GRAD_CLAMP_XFAIL_REASON, strict=False)],
            id="clamp_no_scaled_ckpt",
        ),
    ]

    # Qwen3-235B TP2 per-expert matmuls:
    #   Gate+Up [B, 4096] @ [4096, 1536], Down [B, 768] @ [768, 4096].
    # Use pre-packed expert blocks to cover the matmul shape without routing fanout.
    #
    # Single-block (N==1: block_size == tokens) configs are xfailed — under LNC2
    # SHARD_ON_BLOCK one core is left idle (see SINGLE_BLOCK_XFAIL_REASON). The N==2
    # configs (tokens == 2*block_size) exercise the same shapes with both cores busy.
    qwen3_235B_shapes = [
        # N==2 (both cores get a block): tokens == 2 * block_size.
        pytest.param(4096, 2048, 1, 1024, 1, 768, id="B1024_N2"),
        pytest.param(4096, 4096, 1, 2048, 1, 768, id="B2048_N2"),
        pytest.param(4096, 8192, 1, 4096, 1, 768, id="B4096_N2"),
        # N==1 (single block): xfailed until the col-parallel single-block path lands.
        pytest.param(
            4096,
            1024,
            1,
            1024,
            1,
            768,
            id="B1024",
            marks=pytest.mark.xfail(reason=SINGLE_BLOCK_XFAIL_REASON, strict=False),
        ),
        pytest.param(
            4096,
            2048,
            1,
            2048,
            1,
            768,
            id="B2048",
            marks=pytest.mark.xfail(reason=SINGLE_BLOCK_XFAIL_REASON, strict=False),
        ),
        pytest.param(
            4096,
            4096,
            1,
            4096,
            1,
            768,
            id="B4096",
            marks=pytest.mark.xfail(reason=SINGLE_BLOCK_XFAIL_REASON, strict=False),
        ),
    ]

    @pytest_parametrize(PARAM_NAMES, _build_params(_ALL_FEATURES, _FAST_KEYS_FWD_VALIDATED), abbrevs=_ABBREVS)
    @pytest.mark.parametrize("feature_kwargs", _FEATURE_CONFIGS)
    def test_moe_mxfp8_fwd_validated(
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

    # ------------------------------------------------------------------
    # Test: no_indirect_load single-expert path
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _NO_INDIRECT_LOAD_PARAMS,
    )
    def test_moe_mxfp8_fwd_no_indirect_load(
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
            no_indirect_load=True,
            **feature_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: Qwen3-235B TP2 per-expert forward shapes
    # ------------------------------------------------------------------

    @pytest_parametrize(PARAM_NAMES, qwen3_235B_shapes, abbrevs=_ABBREVS)
    def test_moe_mxfp8_fwd_qwen3_235B_shapes(
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
            no_indirect_load=True,
            use_scale_packing=True,
            spill_reload=True,
            # Skip both activation checkpoints: this suite profiles the FFN math for an
            # apples-to-apples comparison against the bf16 torch-xla baseline (which
            # emits no checkpoints), so the kernel should not spend time on the
            # checkpoint transpose/store/alloc.
            checkpoint_config=MXFP8MOECheckpointConfig(
                save_gate_up_proj_act=False,
                save_scaled_intermediate=False,
            ),
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
        # Exercise both spill on/off; the blocking sweep varies TILES_IN_BLOCK_K,
        # which is what spill_reload reuses across K-blocks.
        "spill_reload",
        [pytest.param(True, id="sr1"), pytest.param(False, id="sr0")],
    )
    def test_moe_mxfp8_fwd_validated_blocking(
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

    # ------------------------------------------------------------------
    # Test: random sweep over all features and blocking params
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
    def test_moe_mxfp8_fwd_validated_sweep(
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
