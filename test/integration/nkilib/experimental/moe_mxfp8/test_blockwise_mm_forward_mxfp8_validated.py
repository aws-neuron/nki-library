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
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import (
    CheckpointLayout,
    MXFP8MOECheckpointConfig,
    checkpoint_block_dims,
)

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
        reuse_spilled_weights: bool = False,
        use_scale_packing: bool = False,
        fast_dma_transpose: bool = False,
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
            "reuse_spilled_weights": reuse_spilled_weights,
            "use_scale_packing": use_scale_packing,
            "fast_dma_transpose": fast_dma_transpose,
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
        N_blocks = self._n_blocks(T, top_k, E, block_size)
        if checkpoint_config.save_gate_up_proj_act:
            gate_up_ckpt_shape = (N_blocks, 2) + checkpoint_block_dims(
                checkpoint_config.gate_up_proj_act_layout, I_TP, block_size
            )
            output_shapes["gate_up_proj_act_checkpoint_T"] = (gate_up_ckpt_shape, bfloat16)
        if checkpoint_config.save_scaled_intermediate:
            scaled_ckpt_shape = (N_blocks,) + checkpoint_block_dims(
                checkpoint_config.scaled_intermediate_layout, I_TP, block_size
            )
            output_shapes["scaled_intermediate_checkpoint_T"] = (scaled_ckpt_shape, bfloat16)

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
        # Odd-N (N = T // B odd) exercises the LNC2 split-tail path on the
        # contiguous single-expert path. B>=256 splits the tail across cores;
        # B=128 falls back to plain round-robin (B/2=64 not TILE_M-aligned).
        pytest.param(1024, 1280, 1, 256, 1, 384, {}, marks=pytest.mark.fast, id="b256_N5_split"),
        pytest.param(2048, 1536, 1, 512, 1, 256, {"spill_reload": True, "use_scale_packing": True}, id="b512_N3_split"),
        pytest.param(1024, 1152, 1, 128, 1, 256, {}, id="b128_N9_no_split"),
        # reuse_spilled_weights: carry the quantized weight spill buffers across the
        # block loop so only the first block each core owns loads+quantizes the
        # weights. Requires no_indirect_load (E=1) and a phase that spills. Each
        # config MUST match its reuse-off counterpart exactly (reuse replays the
        # spilled bytes, so a delta is a real bug). Auto-gen tiling puts these on
        # NUM_M_BLOCKS == NUM_N_BLOCKS == 1, so the in-block reload never fires and
        # cross-block reuse is what makes spilling pay off; see the blocking test for
        # the multi-M-block case.
        #
        # N=4 -> 2 blocks per core, so the second block reads the cache.
        pytest.param(
            1024,
            2048,
            1,
            512,
            1,
            512,
            {"spill_reload": True, "reuse_spilled_weights": True},
            marks=pytest.mark.fast,
            id="b512_N4_reuse",
        ),
        # Odd N=3 + split tail: the tail half-block reads a cache populated by the
        # full-block loop and must not be thrown off by its halved B (weight buffers
        # are sized by H / I_TP only, never B).
        pytest.param(
            2048,
            1536,
            1,
            512,
            1,
            256,
            {"spill_reload": True, "use_scale_packing": True, "reuse_spilled_weights": True},
            id="b512_N3_split_reuse",
        ),
        # fast_dma_transpose: the populating block loads BF16 via the fast 4D DGT
        # path; later blocks skip it. Guards against the fast-DGT flag leaking onto
        # the (quantized) cache TD, where it does not apply.
        pytest.param(
            1024,
            2048,
            1,
            512,
            1,
            512,
            {"spill_reload": True, "fast_dma_transpose": True, "reuse_spilled_weights": True},
            id="b512_N4_reuse_fastdgt",
        ),
        # I_TP=384 < tile_n: auto-gen picks tile_n=128, so the gate/up N axis is
        # tiled 3-ways within one N-block -- a different cache F geometry than above.
        pytest.param(
            1024,
            2048,
            1,
            512,
            1,
            384,
            {"spill_reload": True, "reuse_spilled_weights": True},
            id="b512_N4_reuse_small_tile_n",
        ),
        # N=8 -> 4 blocks per core: the cache is read three times, and B=128 makes
        # each block a single M tile, so the weight load/quantize is the dominant
        # term reuse removes.
        pytest.param(
            512,
            1024,
            1,
            128,
            1,
            256,
            {"spill_reload": True, "reuse_spilled_weights": True},
            id="b128_N8_reuse",
        ),
    ]

    # Qwen3-235B per-expert matmuls, TP2 (I_TP = moe_intermediate_size / 2 = 768)
    # and TP1 (intermediate unsharded, I_TP = 1536):
    #   TP2: Gate+Up [B, 4096] @ [4096, 1536], Down [B, 768]  @ [768, 4096].
    #   TP1: Gate+Up [B, 4096] @ [4096, 3072], Down [B, 1536] @ [1536, 4096].
    # Only I_TP differs between the two, so both share one param list.
    # Use pre-packed expert blocks to cover the matmul shape without routing fanout.
    #
    # Single-block (N==1: block_size == tokens) configs are xfailed — under LNC2
    # SHARD_ON_BLOCK one core is left idle (see SINGLE_BLOCK_XFAIL_REASON). The N==2
    # configs (tokens == 2*block_size) exercise the same shapes with both cores busy.
    qwen3_235B_shapes = [
        pytest.param(4096, tokens, 1, block_size, 1, intermediate, id=f"{tp}_B{block_size}{suffix}", marks=marks)
        for tp, intermediate in (("tp2", 768), ("tp1", 1536))
        for block_size in (1024, 2048, 4096)
        # N==2 (both cores get a block): tokens == 2 * block_size. N==1 (single
        # block) is xfailed until the col-parallel single-block path lands.
        for tokens, suffix, marks in (
            (2 * block_size, "_N2", []),
            (block_size, "", [pytest.mark.xfail(reason=SINGLE_BLOCK_XFAIL_REASON, strict=False)]),
        )
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
    # Odd-N shapes: exercise the LNC2 split-tail load-balancing path.
    # ------------------------------------------------------------------
    # When N (block count) is odd under LNC2 SHARD_ON_BLOCK, the kernel splits
    # the last block along B across the two cores (core 0 -> [0:B/2], core 1 ->
    # [B/2:B]) instead of leaving one core idle. The split needs B/2 to stay
    # TILE_M(128)-aligned (B >= 256); for B=128 the split is skipped and the
    # plain round-robin still covers all N blocks (correctness fallback). These
    # shapes all yield odd N (verified via _n_blocks) so the tail path is hit
    # deterministically; the output and both checkpoints must still pass the
    # shared three-metric MXFP8 validator against the golden.
    _ODD_N_SHAPES = [
        # H,    T,     E, B,    TOPK, I_TP    -> N (odd)
        pytest.param(1024, 1024, 2, 512, 2, 512, marks=pytest.mark.fast, id="B512_N5_split"),
        pytest.param(384, 1024, 4, 512, 1, 256, id="B512_N5_split_e4"),
        pytest.param(2048, 1024, 4, 256, 2, 1024, id="B256_N11_split"),
        pytest.param(1024, 1024, 2, 256, 2, 256, id="B256_N9_split"),
        # B=128: odd N but split skipped (B/2=64 not TILE_M-aligned) -> fallback.
        pytest.param(1024, 1024, 2, 128, 1, 256, id="B128_N9_no_split"),
    ]

    @pytest_parametrize(PARAM_NAMES, _ODD_N_SHAPES, abbrevs=_ABBREVS)
    def test_moe_mxfp8_fwd_odd_n_split_tail(
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
        # Sanity-guard the fixture: these must be genuinely odd N or the test
        # would silently stop covering the split-tail path if a shape changes.
        assert self._n_blocks(tokens, top_k, expert, block_size) % 2 == 1
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

    # Checkpoint-store variants for the profiling comparison: measure the FFN math
    # alone (no checkpoints) against the save layout. Everything else
    # (no_indirect_load, scale_packing, spill_reload, fast_dma_transpose) is held
    # fixed so the only variable is the per-block checkpoint store work.
    #   - ckpt_none: skip both checkpoints -> apples-to-apples vs the bf16 baseline
    #     (which emits none); the kernel spends no time on the store/alloc.
    #   - ckpt_direct: save both, DIRECT (token-major [.., B, I_TP]) -> one plain DMA
    #     per block, no PE transpose.
    # TRANSPOSED is not currently implemented by the block-granular store (see
    # CheckpointLayout), so it has no variant here.
    _QWEN3_CHECKPOINT_CONFIGS = [
        pytest.param(
            MXFP8MOECheckpointConfig(save_gate_up_proj_act=False, save_scaled_intermediate=False),
            id="ckpt_none",
        ),
        pytest.param(
            MXFP8MOECheckpointConfig(
                gate_up_proj_act_layout=CheckpointLayout.DIRECT,
                scaled_intermediate_layout=CheckpointLayout.DIRECT,
            ),
            id="ckpt_direct",
        ),
    ]

    # ------------------------------------------------------------------
    # Test: Qwen3-235B TP2/TP1 per-expert forward shapes
    # ------------------------------------------------------------------

    @pytest_parametrize(PARAM_NAMES, qwen3_235B_shapes, abbrevs=_ABBREVS)
    @pytest.mark.parametrize("checkpoint_config", _QWEN3_CHECKPOINT_CONFIGS)
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
        checkpoint_config: MXFP8MOECheckpointConfig,
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
            # Load the unswizzled-BF16 operands via the fast direct-4D DGT path
            # (valid here because no_indirect_load carries no per-expert offset).
            fast_dma_transpose=True,
            checkpoint_config=checkpoint_config,
        )

    # ------------------------------------------------------------------
    # Test: Qwen3-235B shapes with cross-block weight reuse
    # ------------------------------------------------------------------

    # The qwen3_235B_shapes above are all N==2 (one block per LNC2 core), where reuse
    # cannot fire. These use tokens == 4 * block_size -> N==4 -> 2 blocks per core,
    # the smallest case that exercises reuse and the deployment shape (8192 tokens at
    # B=2048). Otherwise identical to the run above, so reuse on vs off is a clean A/B.
    _QWEN3_REUSE_SHAPES = [
        pytest.param(4096, 4 * block_size, 1, block_size, 1, intermediate, id=f"{tp}_B{block_size}_N4")
        for tp, intermediate in (("tp2", 768), ("tp1", 1536))
        for block_size in (1024, 2048)
    ]

    @pytest_parametrize(PARAM_NAMES, _QWEN3_REUSE_SHAPES, abbrevs=_ABBREVS)
    @pytest.mark.parametrize(
        "reuse_spilled_weights",
        [pytest.param(True, id="reuse1"), pytest.param(False, id="reuse0")],
    )
    def test_moe_mxfp8_fwd_qwen3_235B_reuse_spilled_weights(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        reuse_spilled_weights: bool,
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
            reuse_spilled_weights=reuse_spilled_weights,
            fast_dma_transpose=True,
            # Match the benchmark checkpoint policy: gate/up only, DIRECT layout
            # (plain DMA, no PE transpose), so the FFN cost dominates the measurement.
            checkpoint_config=MXFP8MOECheckpointConfig(
                save_scaled_intermediate=False,
                gate_up_proj_act_layout=CheckpointLayout.DIRECT,
            ),
        )

    # ------------------------------------------------------------------
    # Test: DIRECT (non-transposed) checkpoint store layout
    # ------------------------------------------------------------------

    # DIRECT is the only implemented layout (see CheckpointLayout), and it is now the
    # config default, so per-checkpoint layout combinations would all be the same
    # config. What is still worth covering is which checkpoints are saved, since each
    # save flag gates its own block-granular store.
    _CHECKPOINT_LAYOUT_CONFIGS = [
        pytest.param(
            MXFP8MOECheckpointConfig(
                gate_up_proj_act_layout=CheckpointLayout.DIRECT,
                scaled_intermediate_layout=CheckpointLayout.DIRECT,
            ),
            id="both_direct",
        ),
        pytest.param(
            MXFP8MOECheckpointConfig(save_scaled_intermediate=False),
            id="gate_up_only",
        ),
        pytest.param(
            MXFP8MOECheckpointConfig(save_gate_up_proj_act=False),
            id="scaled_only",
        ),
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize("checkpoint_config", _CHECKPOINT_LAYOUT_CONFIGS)
    def test_moe_mxfp8_fwd_checkpoint_layout(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        checkpoint_config: MXFP8MOECheckpointConfig,
    ):
        # Small no_indirect_load shape; validates the DIRECT store layout against the
        # golden (which mirrors the layout). The golden and output_shapes follow the
        # config, so a disabled checkpoint is simply absent from both.
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=128,
            tokens=1024,
            expert=1,
            block_size=512,
            top_k=1,
            intermediate=128,
            no_indirect_load=True,
            checkpoint_config=checkpoint_config,
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

    # -----------------------------------------------------------------------------------
    # Test: cross-block weight reuse with explicit blocking (both reuse paths live)
    # -----------------------------------------------------------------------------------

    # reuse_spilled_weights requires no_indirect_load (E=1), which the E>1
    # BLOCKING_TEST_PARAMS cannot express, so these carry their own shapes. Explicit
    # blocking drives NUM_M_BLOCKS > 1, so the *in-block* weight reload fires on the
    # populating block while later blocks take the cross-block path -- the pairing
    # that could break the RHS load tile shape or bd, which auto-gen tiling never hits.
    # fmt: off
    _REUSE_BLOCKING_PARAMS = [
        # H,    T,    E, B,   TOPK, I_TP,  tm, tn, tk
        [1024, 2048, 1, 512, 1,    1024,   2,  2,  2],
        [2048, 2048, 1, 512, 1,    1024,   1,  1,  2],
        [1024, 2048, 1, 512, 1,    1024,   1,  1,  1],
        [4096, 2048, 1, 512, 1,    384,    2,  4,  4],
        [2048, 1024, 1, 256, 1,    1024,   4,  4,  8],
    ]
    # fmt: on

    @pytest_parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, tiles_m, tiles_n, tiles_k",
        _REUSE_BLOCKING_PARAMS,
        abbrevs={**_ABBREVS, "tiles_m": "tm", "tiles_n": "tn", "tiles_k": "tk"},
    )
    @pytest.mark.parametrize(
        # Reuse on vs off at identical blocking. The two must agree numerically:
        # reuse replays the exact bytes the spill wrote, so a delta is a real bug.
        "reuse_spilled_weights",
        [pytest.param(True, id="reuse1"), pytest.param(False, id="reuse0")],
    )
    def test_moe_mxfp8_fwd_reuse_spilled_weights_blocking(
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
        reuse_spilled_weights: bool,
    ):
        """Cross-block weight reuse under explicit multi-block blocking."""
        blocking = _compute_blocking_params(
            H=hidden,
            B=block_size,
            I_TP=intermediate,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_k=tiles_k,
        )
        # Sanity-guard the fixture: reuse only does something when this core owns
        # more than one block, so N // num_shards(2) must exceed 1.
        assert self._n_blocks(tokens, top_k, expert, block_size) // 2 > 1
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
            no_indirect_load=True,
            spill_reload=True,
            reuse_spilled_weights=reuse_spilled_weights,
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
