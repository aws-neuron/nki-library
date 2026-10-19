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
checkpoint (gate_up_proj_act_checkpoint_T).
"""

import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, final

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
    _validate_kernel_options,
    blockwise_mm_fwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    blockwise_mm_fwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import (
    CheckpointLayout,
    checkpoint_block_dims,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    TensorDescriptor,
    TensorOrientation,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.dyn_blocks import resolve_blocks, tail_sizes_for

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


# I_TP=1280 with BLOCK_N=1024 leaves a 256-wide final N-block for the
# K_BY_F gate/up free-axis slice.
_KBYF_RAGGED_BLOCKING = _compute_blocking_params(H=1024, B=512, I_TP=1280, tiles_m=1, tiles_n=2, tiles_k=1)


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

            # Independently randomize whether the gate/up checkpoint is saved and its
            # store layout so the sweep exercises the skip path (checkpoint not
            # computed/stored/returned) and both DIRECT/TRANSPOSED layouts under random
            # shapes. Drawn last so the existing feature/tile draws (and thus test ids)
            # stay stable.
            save_gate_up_proj_act = rng.choice([True, False])
            gate_up_proj_act_layout = rng.choice([CheckpointLayout.DIRECT, CheckpointLayout.TRANSPOSED])

            test_id = (
                f"hid_{H}_tok_{T}_exp_{E}_bs_{B}_k_{top_k}_int_{I_TP}"
                f"_sr{int(spill_reload)}_sp{int(use_scale_packing)}"
                f"_b{int(bias)}_pq{int(prequantize_weights)}"
                f"_tm{tiles_m}_tn{tiles_n}_tk{tiles_k}"
                f"_cg{int(save_gate_up_proj_act)}_cl{gate_up_proj_act_layout.name[0]}"
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
                        "save_gate_up_proj_act": save_gate_up_proj_act,
                        "gate_up_proj_act_layout": gate_up_proj_act_layout,
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
    weights_k_by_f=False,
):
    gate_up_shape = (expert, hidden, 2, intermediate) if weights_k_by_f else (expert, intermediate, 2, hidden)
    down_shape = (expert, intermediate, hidden) if weights_k_by_f else (expert, hidden, intermediate)
    return {
        "hidden_states": _fake_tensor((tokens, hidden)),
        "gate_up_proj_weight": _fake_tensor(gate_up_shape),
        "down_proj_weight": _fake_tensor(down_shape),
        "token_position_to_id": _fake_tensor(token_position_shape, dtype=nl.int32),
        "block_to_expert": _fake_tensor(block_to_expert_shape, dtype=nl.int32),
        "expert_affinities_masked": _fake_tensor((tokens * expert, 1)),
        "block_size": block_size,
        "single_expert_dense": True,
        "gate_up_weight_orientation": (TensorOrientation.K_BY_F if weights_k_by_f else TensorOrientation.F_BY_K),
        "down_weight_orientation": TensorOrientation.K_BY_F if weights_k_by_f else TensorOrientation.F_BY_K,
    }


# ============================================================================
# single_expert_dense input contract tests
# ============================================================================


@pytest.mark.fast
def test_moe_mxfp8_fwd_single_expert_dense_builder_contract():
    inputs = build_mxfp8_moe_fwd_inputs(
        tokens=256,
        hidden=128,
        intermediate=128,
        expert=1,
        block_size=128,
        top_k=1,
        single_expert_dense=True,
    )

    config = inputs["config"]
    assert config.single_expert_dense
    assert inputs["token_position_to_id"].shape == (1,)
    assert inputs["block_to_expert"].shape == (1, 1)
    assert inputs["gate_up_proj_weight"].shape[0] == 1
    assert inputs["down_proj_weight"].shape[0] == 1
    assert inputs["expert_affinities_masked"].shape == (256, 1)
    assert not config.is_tensor_update_accumulating
    assert not config.skip_dma.skip_token
    assert not config.skip_dma.skip_weight


@pytest.mark.fast
@pytest.mark.parametrize(
    "override_kwargs, match",
    [
        pytest.param({"expert": 2}, "expert=1", id="multiple_experts"),
        pytest.param({"top_k": 2}, "top_k=1", id="top_k_gt_one"),
        pytest.param({"tokens": 768, "block_size": 512}, "tokens divisible by block_size", id="partial_block"),
    ],
)
def test_moe_mxfp8_fwd_single_expert_dense_input_contract(override_kwargs, match):
    kwargs = {
        "tokens": 1024,
        "hidden": 128,
        "intermediate": 128,
        "expert": 1,
        "block_size": 128,
        "top_k": 1,
        "single_expert_dense": True,
    }
    kwargs.update(override_kwargs)
    with pytest.raises(AssertionError, match=match):
        build_mxfp8_moe_fwd_inputs(**kwargs)


@pytest.mark.fast
@pytest.mark.parametrize(
    "N, num_static_block, has_dynamic, expected",
    [
        # No token count -> fully static, whatever was requested.
        pytest.param(8, None, False, (8, 0), id="no_dynamic"),
        pytest.param(8, 4, False, (8, 0), id="no_dynamic_with_value"),
        # None -> auto: N - num_shards (mirrors the bf16 kernel's N - E).
        pytest.param(8, None, True, (6, 2), id="auto_n8"),
        pytest.param(4, None, True, (2, 2), id="auto_n4"),
        # Auto floors at min(num_shards, N), so short schedules remain in bounds.
        pytest.param(2, None, True, (2, 0), id="auto_n2_floor"),
        pytest.param(1, None, True, (1, 0), id="auto_n1_clamped_to_capacity"),
        pytest.param(3, None, True, (2, 1), id="auto_n3"),
        # Explicit values are clamped to [min(num_shards, N), N]; parity is preserved
        # because an odd static count splits its last block across the cores.
        pytest.param(8, 2, True, (2, 6), id="explicit_min"),
        pytest.param(8, 3, True, (3, 5), id="explicit_odd_kept"),
        pytest.param(8, 5, True, (5, 3), id="explicit_odd_both_odd"),
        pytest.param(8, 8, True, (8, 0), id="explicit_full"),
        pytest.param(8, 99, True, (8, 0), id="explicit_over_n"),
        pytest.param(8, 1, True, (2, 6), id="explicit_below_num_shards"),
        pytest.param(8, 0, True, (2, 6), id="explicit_zero"),
        pytest.param(1, 0, True, (1, 0), id="explicit_n1_clamped_to_capacity"),
    ],
)
def test_moe_mxfp8_fwd_resolve_static_blocks(N, num_static_block, has_dynamic, expected):
    schedule = resolve_blocks(N, block_size=512, num_static_block=num_static_block, has_dynamic=has_dynamic)
    assert (schedule.n_static, schedule.n_dynamic) == expected


@pytest.mark.fast
@pytest.mark.parametrize(
    "N, num_static_block, can_split_tail, expected",
    [
        # Even static prefix: nothing to halve, the loop walks every static block.
        pytest.param(8, 4, True, (4, False, True), id="even_static"),
        # Odd static prefix: the last block is held back for the half-split.
        pytest.param(8, 3, True, (2, True, True), id="odd_static_splits"),
        pytest.param(8, 5, True, (4, True, True), id="odd_static_splits_5"),
        pytest.param(1, 0, True, (0, True, True), id="single_block_splits"),
        # block_size // 2 not tile-aligned: neither region can half-split, so the
        # static loop walks all of them and the dynamic region takes the clamped path.
        pytest.param(8, 3, False, (3, False, False), id="no_split_possible"),
        # No token count at all: fully static, so nothing is held back — but the
        # capability flag still reports what the caller passed.
        pytest.param(8, None, True, (8, False, True), id="static_only"),
    ],
)
def test_moe_mxfp8_fwd_resolve_blocks_tail_split(N, num_static_block, can_split_tail, expected):
    schedule = resolve_blocks(
        N,
        block_size=512,
        num_static_block=num_static_block,
        has_dynamic=num_static_block is not None,
        can_split_halves=can_split_tail,
    )
    assert (schedule.static_whole, schedule.split_static_tail, schedule.can_split_halves) == expected


@pytest.mark.fast
@pytest.mark.parametrize(
    "block_size, expected",
    [
        # Only sizes strictly below block_size are useful; block_size always terminates
        # the list so a trailing block is never longer than a whole one.
        pytest.param(4096, [1024, 2048, 4096], id="b4096_all_three"),
        pytest.param(2048, [1024, 2048], id="b2048_two"),
        pytest.param(1024, [1024], id="b1024_itself"),
        pytest.param(512, [512], id="b512_no_shortening"),
        pytest.param(128, [128], id="b128_no_shortening"),
        pytest.param(8192, [1024, 2048, 4096, 8192], id="b8192_adds_full"),
    ],
)
def test_moe_mxfp8_fwd_tail_sizes_for(block_size, expected):
    assert tail_sizes_for(block_size) == expected


@pytest.mark.fast
@pytest.mark.parametrize(
    "block_size, remainder, expected",
    [
        # The selected size is the smallest that covers the remainder.
        pytest.param(4096, 1, 1024, id="one_token"),
        pytest.param(4096, 300, 1024, id="small"),
        pytest.param(4096, 1024, 1024, id="exactly_1024"),
        pytest.param(4096, 1025, 2048, id="just_over_1024"),
        pytest.param(4096, 1500, 2048, id="mid"),
        pytest.param(4096, 2048, 2048, id="exactly_2048"),
        pytest.param(4096, 3000, 4096, id="large"),
        pytest.param(2048, 1500, 2048, id="b2048_mid"),
        pytest.param(512, 100, 512, id="b512_cannot_shorten"),
    ],
)
def test_moe_mxfp8_fwd_tail_size_selection(block_size, remainder, expected):
    # Mirrors the device selector: the first candidate >= the remainder wins.
    sizes = tail_sizes_for(block_size)
    assert next(size for size in sizes if remainder <= size) == expected


@pytest.mark.fast
def test_moe_mxfp8_fwd_resolve_blocks_rejects_non_power_of_two_block():
    with pytest.raises(AssertionError, match="block_size must be a power of two"):
        resolve_blocks(4, block_size=768, num_static_block=2, has_dynamic=True)


@pytest.mark.fast
def test_moe_mxfp8_fwd_num_valid_tokens_requires_single_expert_dense():
    config = MXFP8MOEFwdConfig(single_expert_dense=False)
    with pytest.raises(AssertionError, match="requires single_expert_dense"):
        _validate_kernel_options(config=config, num_valid_tokens=_fake_tensor((1,), nl.int32))


@pytest.mark.fast
@pytest.mark.parametrize(
    "shape, dtype, match",
    [
        pytest.param((8,), nl.int32, r"must be \[1\]", id="wrong_length"),
        pytest.param((1, 1), nl.int32, r"must be \[1\]", id="wrong_rank"),
        pytest.param((1,), nl.float32, "must be int32", id="wrong_dtype"),
    ],
)
def test_moe_mxfp8_fwd_num_valid_tokens_shape_contract(shape, dtype, match):
    kwargs = _build_no_indirect_validator_inputs(tokens=1024, block_size=128)
    kwargs["num_valid_tokens"] = _fake_tensor(shape, dtype)
    with pytest.raises(AssertionError, match=match):
        _validate_inputs_and_derive_dims(**kwargs)


@pytest.mark.fast
def test_moe_mxfp8_fwd_num_valid_tokens_accepts_scalar():
    kwargs = _build_no_indirect_validator_inputs(tokens=1024, block_size=128)
    kwargs["num_valid_tokens"] = _fake_tensor((1,), nl.int32)
    assert _validate_inputs_and_derive_dims(**kwargs)[4] == 8


@pytest.mark.fast
def test_moe_mxfp8_fwd_num_valid_tokens_allows_single_dynamic_block():
    kwargs = _build_no_indirect_validator_inputs(tokens=512, block_size=512)
    kwargs["num_valid_tokens"] = _fake_tensor((1,), nl.int32)
    assert _validate_inputs_and_derive_dims(**kwargs)[4] == 1


@pytest.mark.fast
def test_moe_mxfp8_fwd_single_expert_dense_validator_derives_contiguous_blocks():
    assert _validate_inputs_and_derive_dims(**_build_no_indirect_validator_inputs(tokens=512, block_size=256)) == (
        512,
        128,
        128,
        1,
        2,
    )


@pytest.mark.fast
def test_moe_mxfp8_fwd_kbyf_validator_derives_dims():
    assert _validate_inputs_and_derive_dims(
        **_build_no_indirect_validator_inputs(
            tokens=512,
            block_size=256,
            hidden=128,
            intermediate=256,
            weights_k_by_f=True,
        )
    ) == (512, 128, 256, 1, 2)


@pytest.mark.fast
def test_moe_mxfp8_fwd_kbyf_rejects_forward_natural_shapes():
    kwargs = _build_no_indirect_validator_inputs(
        tokens=512,
        block_size=256,
        hidden=128,
        intermediate=256,
    )
    kwargs["gate_up_weight_orientation"] = TensorOrientation.K_BY_F
    kwargs["down_weight_orientation"] = TensorOrientation.K_BY_F
    with pytest.raises(AssertionError, match="must match backward-natural"):
        _validate_inputs_and_derive_dims(**kwargs)


def test_moe_mxfp8_fwd_kbyf_builder_emits_backward_natural_weights():
    inputs = build_mxfp8_moe_fwd_inputs(
        tokens=512,
        hidden=384,
        intermediate=256,
        expert=1,
        block_size=256,
        top_k=1,
        single_expert_dense=True,
        weights_k_by_f=True,
    )
    assert inputs["gate_up_proj_weight"].shape == (1, 384, 2, 256)
    assert inputs["down_proj_weight"].shape == (1, 256, 384)
    assert inputs["config"].gate_up_weight_td.orientation == TensorOrientation.K_BY_F
    assert inputs["config"].down_weight_td.orientation == TensorOrientation.K_BY_F


@pytest.mark.fast
@pytest.mark.parametrize(
    "override_kwargs, match",
    [
        pytest.param(
            {
                "gate_up_proj_weight": _fake_tensor((2, 128, 2, 128)),
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
def test_moe_mxfp8_fwd_single_expert_dense_kernel_contract(override_kwargs, match):
    kwargs = _build_no_indirect_validator_inputs()
    kwargs.update(override_kwargs)
    with pytest.raises(AssertionError, match=match):
        _validate_inputs_and_derive_dims(**kwargs)


@pytest.mark.fast
def test_moe_mxfp8_fwd_prequant_mismatched_weight_scheme_rejected():
    # gate/up and down share one x4 swizzle scheme (it drives the single on-the-fly
    # activation load); supplying weight TDs whose quant_scheme disagrees must fail loudly.
    # The entry resolves the scheme (and asserts) before any device work, so a direct call
    # with dummy tensors exercises the check.
    config = MXFP8MOEFwdConfig(
        gate_up_weight_td=TensorDescriptor(quant_scheme=QuantScheme.WRAPX),
        down_weight_td=TensorDescriptor(quant_scheme=QuantScheme._1x32),
    )
    with pytest.raises(AssertionError, match="quant_scheme must match"):
        blockwise_mm_fwd_mxfp8(
            hidden_states=_fake_tensor((256, 128)),
            expert_affinities_masked=_fake_tensor((256, 1)),
            gate_up_proj_weight=_fake_tensor((1, 64, 128)),
            down_proj_weight=_fake_tensor((1, 32, 128)),
            token_position_to_id=_fake_tensor((1,), dtype=nl.int32),
            block_to_expert=_fake_tensor((1, 1), dtype=nl.int32),
            config=config,
            gate_up_weight_scales=_fake_tensor((1, 8, 128)),
            down_weight_scales=_fake_tensor((1, 8, 128)),
        )


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
        use_1x32_pe_swizzle: bool = False,
        bias: bool = False,
        clamp_limits=None,
        prequantize_weights: bool = False,
        save_gate_up_proj_act=True,
        gate_up_proj_act_layout=CheckpointLayout.DIRECT,
        single_expert_dense: bool = False,
        weights_k_by_f: bool = False,
        omit_config: bool = False,
        omit_block_size: bool = False,
        num_valid_tokens: Optional[int] = None,
        num_static_block=None,
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
            "blocking_params": blocking_params,
            "spill_reload": spill_reload,
            "reuse_spilled_weights": reuse_spilled_weights,
            "use_scale_packing": use_scale_packing,
            "fast_dma_transpose": fast_dma_transpose,
            "use_1x32_pe_swizzle": use_1x32_pe_swizzle,
            "prequantize_weights": prequantize_weights,
            "bias": bias,
            "clamp_limits": clamp_limits,
            "save_gate_up_proj_act": save_gate_up_proj_act,
            "gate_up_proj_act_layout": gate_up_proj_act_layout,
            "single_expert_dense": single_expert_dense,
            "weights_k_by_f": weights_k_by_f,
            "num_valid_tokens": num_valid_tokens,
            "num_static_block": num_static_block,
        }

        if prequantize_weights:
            kernel_inputs, orig_gate_up_weight, orig_down_weight = build_mxfp8_moe_fwd_inputs(**build_kwargs)
            # Drop config / block_size so the kernel falls back to its own defaults
            # (config -> MXFP8MOEFwdConfig(), block_size -> 4096). Passing scales still
            # auto-selects the pre-quantized path, and the default config resolves to
            # 1x32 + scale packing on the single_expert_dense path.
            if omit_config:
                del kernel_inputs["config"]
            if omit_block_size:
                del kernel_inputs["block_size"]

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
                inputs = build_mxfp8_moe_fwd_inputs(**build_kwargs)
                # Drop config / block_size so the kernel falls back to its own
                # defaults (config -> MXFP8MOEFwdConfig(), block_size -> 4096). The
                # torch ref mirrors the same defaults, so kernel and golden agree.
                if omit_config:
                    del inputs["config"]
                if omit_block_size:
                    del inputs["block_size"]
                return inputs

            torch_ref = torch_ref_wrapper(blockwise_mm_fwd_mxfp8_torch_ref)

        # Output names match the kernel's positional outputs (and the torch ref's
        # dict keys); each checkpoint is present only when its save flag is set.
        output_shapes = {
            "output_hidden_states": ((T, H), bfloat16),
        }
        N_blocks = self._n_blocks(T, top_k, E, block_size)
        if save_gate_up_proj_act:
            gate_up_ckpt_shape = (N_blocks, 2) + checkpoint_block_dims(gate_up_proj_act_layout, I_TP, block_size)
            output_shapes["gate_up_proj_act_checkpoint_T"] = (gate_up_ckpt_shape, bfloat16)

        def output_tensors(kernel_input):
            return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in output_shapes.items()}

        # Rows past the effective valid-token count are undefined. Checkpoint blocks in
        # the static prefix are always produced; the dynamic prefix includes the occupied
        # blocks, and a shortened final slot is compared in full so its zero-filled
        # remainder remains part of the forward/backward contract.
        active_prefix = None
        if num_valid_tokens is not None:
            effective_valid_tokens = min(num_valid_tokens, T)
            active_prefix = {"output_hidden_states": effective_valid_tokens}
            if save_gate_up_proj_act:
                schedule = resolve_blocks(
                    n_blocks=N_blocks,
                    block_size=block_size,
                    num_static_block=num_static_block,
                    has_dynamic=True,
                )
                dynamic_tokens = max(0, effective_valid_tokens - schedule.n_static * block_size)
                active_dynamic = min(math.ceil(dynamic_tokens / block_size), schedule.n_dynamic)
                active_prefix["gate_up_proj_act_checkpoint_T"] = schedule.n_static + active_dynamic

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
            custom_comparator=_moe_fwd_comparator(output_shapes, active_prefix),
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
        # Pre-quantized weights are NOT folded into this shape spread: _ALL_FEATURES
        # includes square (H == I_TP) shapes, which the pre-quantized down path does
        # not yet handle (see _PREQUANT_NO_INDIRECT_PARAMS). Pre-quantized coverage
        # lives in the dedicated test_moe_mxfp8_fwd_prequantized_single_expert_dense (E=1,
        # the deployment path) and test_moe_mxfp8_fwd_prequantized_routed (E>1) methods.
        pytest.param(
            {"gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED},
            id="transposed_ckpt",
        ),
        pytest.param(
            {"save_gate_up_proj_act": False},
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
            },
            # clamp_limits set -> xfail like every other grad-clamp config (strict=False).
            marks=[pytest.mark.fast, pytest.mark.xfail(reason=GRAD_CLAMP_XFAIL_REASON, strict=False)],
            id="clamp",
        ),
        # Odd-N (N = T // B odd) exercises the LNC2 split-tail path on the
        # contiguous single-expert path. B>=256 splits the tail across cores;
        # B=128 falls back to plain round-robin (B/2=64 not TILE_M-aligned).
        pytest.param(1024, 1280, 1, 256, 1, 384, {}, marks=pytest.mark.fast, id="b256_N5_split"),
        pytest.param(2048, 1536, 1, 512, 1, 256, {"spill_reload": True, "use_scale_packing": True}, id="b512_N3_split"),
        pytest.param(1024, 1152, 1, 128, 1, 256, {}, id="b128_N9_no_split"),
        # reuse_spilled_weights: carry the quantized weight spill buffers across the
        # block loop so only the first block each core owns loads+quantizes the
        # weights. Requires single_expert_dense (E=1) and a phase that spills. Each
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
        # use_1x32_pe_swizzle: load the unswizzled-BF16 operands via the 1x32
        # PE-swizzle transpose (load_tile_PE_Swizzle_1x32) instead of the DGT /
        # fast_dma_transpose path. Gated to single_expert_dense (E=1). The swizzle mapping
        # must match on both operands of each GEMM, so the kernel applies 1x32 to all
        # four operand TDs; these cases validate that end-to-end against the shared
        # MXFP8 three-metric golden.
        #
        # Minimal N=2 baseline, both LNC2 cores busy.
        pytest.param(512, 1024, 1, 512, 1, 512, {"use_1x32_pe_swizzle": True}, marks=pytest.mark.fast, id="b512_1x32"),
        # Multi-K-tile on both GEMMs (H=1024, I_TP=1024): exercises the K-tile loop.
        pytest.param(
            1024,
            1024,
            1,
            512,
            1,
            1024,
            {"use_1x32_pe_swizzle": True},
            marks=pytest.mark.fast,
            id="b512_1x32_multi_ktile",
        ),
        # 1x32 through the packed-scale path (the 1x32 loader feeds the same
        # quantize/scale-pack machinery as DGT).
        pytest.param(
            512,
            1024,
            1,
            512,
            1,
            512,
            {"use_scale_packing": True, "use_1x32_pe_swizzle": True},
            id="b512_1x32_scale_packing",
        ),
        # 1x32 + spill_reload: the populating block loads BF16 via 1x32, quantizes, and
        # spills the swizzled x4; guards against the 1x32 flag leaking onto the
        # (already-quantized) spill/reload TD.
        pytest.param(
            1024, 1024, 1, 512, 1, 512, {"spill_reload": True, "use_1x32_pe_swizzle": True}, id="b512_1x32_spill_reload"
        ),
        # 1x32 + spill_reload + reuse (N=4 -> 2 blocks/core): clone of b512_N4_reuse with
        # the load mode swapped. The reuse block reloads the quantized x4 via dma_copy
        # (quant_scheme is ignored there), so LHS/RHS stay swizzle-consistent.
        pytest.param(
            1024,
            2048,
            1,
            512,
            1,
            512,
            {"spill_reload": True, "reuse_spilled_weights": True, "use_1x32_pe_swizzle": True},
            id="b512_N4_reuse_1x32",
        ),
        # Odd N=3 split-tail: the tail half-block (B/2=256, TILE_M-aligned) runs the 1x32
        # load path too, so all four operand TDs must carry the mode into the half-block.
        pytest.param(1024, 1536, 1, 512, 1, 512, {"use_1x32_pe_swizzle": True}, id="b512_N3_split_1x32"),
        # Larger block_size B=1024 (N=2): tile_f (= B for the activation LHS) at 1024.
        pytest.param(1024, 2048, 1, 1024, 1, 512, {"use_1x32_pe_swizzle": True}, id="b1024_1x32"),
        # I_TP=768 (down K), H=768 (gate/up K), and both at once (e.g. Qwen3 TP2).
        pytest.param(512, 1024, 1, 512, 1, 768, {"use_1x32_pe_swizzle": True}, id="b512_1x32_itp768"),
        pytest.param(768, 1024, 1, 512, 1, 512, {"use_1x32_pe_swizzle": True}, id="b512_1x32_h768"),
        pytest.param(768, 1024, 1, 512, 1, 768, {"use_1x32_pe_swizzle": True}, id="b512_1x32_k768_both"),
        # Non-tile-aligned H (768=512+256) on fast-DGT: the gate K-slice's partial final tile
        # must be zero-masked (_kslice_partial) so its unused partitions don't corrupt the sum.
        pytest.param(
            768, 1024, 1, 512, 1, 512, {"fast_dma_transpose": True}, marks=pytest.mark.fast, id="b512_fastdgt_h768"
        ),
        # Backward-natural BF16 weights: gate/up is sliced on N rather than K.
        # Cover both PE mappings, square-shape orientation ambiguity, ragged K,
        # cached weights, odd-N tail splitting, and a ragged N block.
        pytest.param(
            512,
            1024,
            1,
            512,
            1,
            512,
            {"weights_k_by_f": True, "use_1x32_pe_swizzle": True},
            marks=pytest.mark.fast,
            id="b512_kbyf_1x32",
        ),
        pytest.param(512, 1024, 1, 512, 1, 512, {"weights_k_by_f": True}, id="b512_kbyf_wrapx"),
        pytest.param(
            768,
            2048,
            1,
            512,
            1,
            768,
            {
                "weights_k_by_f": True,
                "use_1x32_pe_swizzle": True,
                "spill_reload": True,
                "reuse_spilled_weights": True,
            },
            id="b512_kbyf_k768_reuse",
        ),
        pytest.param(
            1024,
            1536,
            1,
            512,
            1,
            512,
            {"weights_k_by_f": True, "use_1x32_pe_swizzle": True},
            id="b512_kbyf_N3_split",
        ),
        pytest.param(
            1024,
            1024,
            1,
            512,
            1,
            1280,
            {
                "weights_k_by_f": True,
                "use_1x32_pe_swizzle": True,
                "blocking_params": _KBYF_RAGGED_BLOCKING,
            },
            id="b512_kbyf_ragged_blockn",
        ),
        # --- Hardware dynamic block loop (num_valid_tokens) -----------------------
        # T is a per-expert capacity, so only `active_blocks` of the N blocks hold real
        # tokens; the rest must be skipped. num_static_block is pinned to 2 (the minimum:
        # a multiple of num_shards, and >= num_shards so both cores warm the weight
        # cache) so the dynamic region is large enough to be interesting. Skipped
        # blocks' output rows / checkpoint slots are never written, so the comparator
        # only checks the active prefix.
        #
        # Fully static control for the dyn cells below: same shape with no token count, so
        # every block goes through the statically unrolled loop. b512_N8_dyn_a8_s2 must
        # produce bit-identical output despite routing blocks 2-7 through the fori_loop.
        pytest.param(1024, 4096, 1, 512, 1, 512, {}, id="b512_N8_static_ref"),
        # Odd active count: one whole block per core, then both cores split the last. Also warms the reused weight cache in the static region.
        pytest.param(
            1024,
            4096,
            1,
            512,
            1,
            512,
            {
                "num_valid_tokens": 5 * 512,
                "num_static_block": 2,
                "spill_reload": True,
                "reuse_spilled_weights": True,
            },
            marks=pytest.mark.fast,
            id="b512_N8_dyn_a5_s2",
        ),
        # Even A_dyn=4 -> both cores run 2 dynamic blocks (symmetric control for the above).
        pytest.param(
            1024, 4096, 1, 512, 1, 512, {"num_valid_tokens": 6 * 512, "num_static_block": 2}, id="b512_N8_dyn_a6_s2"
        ),
        # All blocks active: must match the plain static result exactly.
        pytest.param(
            1024, 4096, 1, 512, 1, 512, {"num_valid_tokens": 8 * 512, "num_static_block": 2}, id="b512_N8_dyn_a8_s2"
        ),
        # A_dyn=0 -> the fori_loop must run zero iterations on both cores.
        pytest.param(
            1024, 4096, 1, 512, 1, 512, {"num_valid_tokens": 2 * 512, "num_static_block": 2}, id="b512_N8_dyn_a2_s2"
        ),
        # Odd N=5: exercises the clamp that keeps the last-block index in range.
        pytest.param(
            1024, 1280, 1, 256, 1, 384, {"num_valid_tokens": 3 * 256, "num_static_block": 2}, id="b256_N5_dyn_a3_s2"
        ),
        # TRANSPOSED checkpoint: exercises the runtime block base in _store_block_i_tp_major.
        pytest.param(
            1024,
            4096,
            1,
            512,
            1,
            512,
            {
                "num_valid_tokens": 5 * 512,
                "num_static_block": 2,
                "gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED,
            },
            id="b512_N8_dyn_ckptT",
        ),
        # The two fast LHS load modes over the staged hidden block (they cannot take a
        # runtime row offset, so the dynamic body hands them a standalone [B, H] tensor).
        pytest.param(
            1024,
            4096,
            1,
            512,
            1,
            512,
            {"num_valid_tokens": 5 * 512, "num_static_block": 2, "use_1x32_pe_swizzle": True},
            marks=pytest.mark.fast,
            id="b512_N8_dyn_1x32",
        ),
        pytest.param(
            768,
            4096,
            1,
            512,
            1,
            512,
            {"num_valid_tokens": 5 * 512, "num_static_block": 2, "fast_dma_transpose": True},
            id="b512_N8_dyn_fastdgt_h768",
        ),
        # Odd STATIC count only (n_static=3, A_dyn=2): the last static block is split
        # across the cores, and the dynamic region then starts at an odd boundary.
        pytest.param(
            1024,
            4096,
            1,
            512,
            1,
            512,
            {"num_valid_tokens": 5 * 512, "num_static_block": 3},
            id="b512_N8_dyn_oddstatic_s3",
        ),
        # Odd static AND odd dynamic at once (n_static=3, A_dyn=3): both half-splits fire
        # in the same kernel.
        pytest.param(
            1024,
            4096,
            1,
            512,
            1,
            512,
            {"num_valid_tokens": 6 * 512, "num_static_block": 3},
            id="b512_N8_dyn_oddboth_s3",
        ),
        # Shortened trailing block. block_size=4096 gives all three tail sizes, so each
        # cell lands on a different one: the remainder picks the smallest that covers it.
        # (H=512, I_TP=256, T=16384 -> N=4, num_static_block=2.)
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 2 * 4096 + 300, "num_static_block": 2},
            id="b4096_N4_tail1024",
        ),
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {
                "num_valid_tokens": 2 * 4096 + 300,
                "num_static_block": 2,
                "gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED,
            },
            id="b4096_N4_tail1024_ckptT",
        ),
        # Simulator-sized shortened tails validate the zero-filled checkpoint
        # remainder for both on-HBM layouts.
        pytest.param(
            128,
            6144,
            1,
            2048,
            1,
            128,
            {"num_valid_tokens": 2 * 2048 + 300, "num_static_block": 2},
            marks=pytest.mark.fast,
            id="b2048_N3_tail1024_small",
        ),
        pytest.param(
            128,
            6144,
            1,
            2048,
            1,
            128,
            {
                "num_valid_tokens": 2 * 2048 + 300,
                "num_static_block": 2,
                "gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED,
            },
            marks=pytest.mark.fast,
            id="b2048_N3_tail1024_small_ckptT",
        ),
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 2 * 4096 + 1500, "num_static_block": 2},
            id="b4096_N4_tail2048",
        ),
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 2 * 4096 + 3000, "num_static_block": 2},
            id="b4096_N4_tail4096",
        ),
        # Exact multiple of block_size: the last block is unpaired but entirely real,
        # so the selected size is the full block.
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 3 * 4096, "num_static_block": 2},
            id="b4096_N4_last_full",
        ),
        # An EVEN dynamic block count pairs the partial last block with a full one, so
        # it runs at full size: its core is not the bottleneck, so shortening buys
        # nothing.
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 3 * 4096 + 300, "num_static_block": 2},
            id="b4096_N4_paired_no_shorten",
        ),
        # A remainder of exactly a tail size must pick that size, not the next one up.
        pytest.param(
            512,
            16384,
            1,
            4096,
            1,
            256,
            {"num_valid_tokens": 2 * 4096 + 1024, "num_static_block": 2},
            id="b4096_N4_tail_exact1024",
        ),
        # block_size=2048 has only two tail sizes, and here the partial block lands
        # inside the static prefix (n_full=1 < num_static_block=2), so no tail runs.
        pytest.param(
            512,
            8192,
            1,
            2048,
            1,
            256,
            {"num_valid_tokens": 2048 + 500, "num_static_block": 2},
            id="b2048_N4_tail_in_static",
        ),
        # Trailing block with a remainder smaller than one tile row.
        pytest.param(
            512,
            8192,
            1,
            2048,
            1,
            256,
            {"num_valid_tokens": 2 * 2048 + 17, "num_static_block": 2},
            id="b2048_N4_tail_tiny",
        ),
        # Many small blocks with a large dynamic region (N=32, A_dyn=15).
        pytest.param(
            512,
            4096,
            1,
            128,
            1,
            256,
            {
                "num_valid_tokens": 17 * 128,
                "num_static_block": 2,
                "spill_reload": True,
                "reuse_spilled_weights": True,
            },
            id="b128_N32_dyn_a17_s2",
        ),
        # A dynamic single-block schedule clamps its static prefix to N=1 instead of
        # addressing block 1. The odd prefix is split across the two cores.
        pytest.param(
            512,
            512,
            1,
            512,
            1,
            256,
            {"num_valid_tokens": 512, "num_static_block": 0},
            marks=pytest.mark.fast,
            id="b512_N1_dyn",
        ),
        # Runtime counts above capacity are clamped to T, so every block runs once.
        pytest.param(
            128,
            2048,
            1,
            512,
            1,
            128,
            {"num_valid_tokens": 5 * 512, "num_static_block": 2},
            marks=pytest.mark.fast,
            id="b512_N4_dyn_over_capacity_small",
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
    # Test: single_expert_dense single-expert path
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _NO_INDIRECT_LOAD_PARAMS,
    )
    def test_moe_mxfp8_fwd_single_expert_dense(
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
            single_expert_dense=True,
            **feature_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: framework-style call with no config and no block_size (defaults)
    # ------------------------------------------------------------------

    def test_moe_mxfp8_fwd_defaults_omit_config_and_block_size(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
    ):
        """Kernel runs on its own defaults when config + block_size are omitted.

        Omitting ``config`` -> ``MXFP8MOEFwdConfig()`` and omitting ``block_size`` ->
        4096 inside the kernel (the framework's minimal-setup call). The default config
        is the single-expert dense path (spill/reload, weight reuse and the 1x32
        PE-swizzle load all default on), so the inputs are built dense: E=1, top_k=1,
        T == N * block_size, and dummy routing tensors. The torch ref resolves the same
        defaults, so kernel and golden agree.
        """
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=256,
            tokens=8192,
            expert=1,
            block_size=4096,
            top_k=1,
            intermediate=256,
            single_expert_dense=True,
            omit_config=True,
            omit_block_size=True,
        )

    # ------------------------------------------------------------------
    # Test: pre-quantized weights on the contiguous single-expert (E=1) path.
    # ------------------------------------------------------------------
    # This is the deployment/inference use case: weights are quantized ONCE, offline,
    # into the forward-natural MXFP8 x4 layout (gate_up [E, H/4, 2*I_TP], down
    # [E, I_TP/4, H]) plus companion MX scales; the kernel skips the on-chip weight
    # quantize + spill and consumes the x4 data + scales directly. The activation
    # (LHS) is still quantized per-block on-chip, via the WRAPX (DGT / fast_dma_transpose)
    # swizzle that matches the offline weight swizzle. Every cell is checked against
    # the three-metric MXFP8 golden fed the original fp32 weights (via _pq_torch_ref).
    # Covers packed scales, multi-K-tile, non-512 dims on BOTH axes (gate/up H and down
    # I_TP in {128, 384, 768}), fast_dma_transpose, spill_reload (spills only the quantized
    # activation), the odd-N split tail, larger block sizes, and the Qwen3-235B TP1/TP2
    # per-expert shapes.
    #
    # NOTE: every shape keeps H != I_TP (a logically-square down weight is a follow-up).
    # Non-512 H is supported: on the E=1 path the x4 gate/up weight is sliced into two
    # standalone K=H views, so each GEMM is an ordinary non-512-aligned-K x4 matmul whose
    # partial last tile is masked by the existing path -- H = 128 / 768 work as well as the
    # 512-aligned shapes.
    _PREQUANT_NO_INDIRECT_PARAMS = [
        # H,    T,    E, B,    TOPK, I_TP, feature_kwargs
        # NOTE: all cases use packed scales (use_scale_packing=True), the standard MXFP8
        # deployment layout. Unpacked pre-quantized scales are a separate follow-up: the
        # oversized-2D golden emits raw uint8 e8m0 bits, which the matmul rejects against
        # the on-the-fly activation's float8_e8m0fnu scales (and the backward does not
        # exercise unpacked pre-quantized weights either).
        pytest.param(128, 1024, 1, 128, 1, 256, {"use_scale_packing": True}, marks=pytest.mark.fast, id="pq_b128"),
        # Small I_TP=256 with B=512.
        pytest.param(
            512, 1024, 1, 512, 1, 256, {"use_scale_packing": True}, marks=pytest.mark.fast, id="pq_b512_small"
        ),
        # Multi-K-tile on both GEMMs (H=1024 -> 2 K-tiles, I_TP=1536 -> 3 K-tiles).
        pytest.param(
            1024, 1024, 1, 512, 1, 1536, {"use_scale_packing": True}, marks=pytest.mark.fast, id="pq_multi_ktile"
        ),
        # I_TP=384 < tile_n -> gate/up N tiled 3-ways (different scale F geometry).
        pytest.param(1024, 2048, 1, 512, 1, 384, {"use_scale_packing": True}, id="pq_itp384"),
        # Non-512 gate/up K (H=768 -> gate slice [0,768) = 512+256, a mid-tile K-slice end)
        # and non-512 down K (I_TP=768). Exercises the quantized K-slice boundary clamp.
        pytest.param(768, 1024, 1, 512, 1, 512, {"use_scale_packing": True}, id="pq_h768"),
        pytest.param(512, 1024, 1, 512, 1, 768, {"use_scale_packing": True}, id="pq_itp768"),
        # fast_dma_transpose on the activation LHS (WRAPX, matches the x4 weight swizzle).
        pytest.param(
            1024, 2048, 1, 512, 1, 512, {"use_scale_packing": True, "fast_dma_transpose": True}, id="pq_fastdgt"
        ),
        # spill_reload: weights are already x4 (no weight spill); only the quantized
        # activation is spilled/reloaded, so this guards the is_quantized spill bypass.
        pytest.param(1024, 2048, 1, 512, 1, 512, {"use_scale_packing": True, "spill_reload": True}, id="pq_spill"),
        # Odd N=3 split tail: the tail half-block also runs the prequant load path.
        pytest.param(1024, 1536, 1, 512, 1, 512, {"use_scale_packing": True}, id="pq_N3_split"),
        # Larger block sizes.
        pytest.param(1024, 2048, 1, 1024, 1, 512, {"use_scale_packing": True}, id="pq_b1024"),
        pytest.param(2048, 4096, 1, 2048, 1, 512, {"use_scale_packing": True}, id="pq_b2048"),
        # Qwen3-235B per-expert shapes (H=4096), TP1 (I_TP=1536) and TP2 (I_TP=768),
        # N=2 (both LNC2 cores busy).
        pytest.param(4096, 4096, 1, 2048, 1, 1536, {"use_scale_packing": True}, id="pq_qwen3_tp1"),
        pytest.param(4096, 4096, 1, 2048, 1, 768, {"use_scale_packing": True}, id="pq_qwen3_tp2"),
    ]

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _PREQUANT_NO_INDIRECT_PARAMS,
    )
    def test_moe_mxfp8_fwd_prequantized_single_expert_dense(
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
            single_expert_dense=True,
            prequantize_weights=True,
            **feature_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: pre-quantized weights on the ROUTED (E>1, indirect-load) path.
    # ------------------------------------------------------------------
    # Exercises the per-expert slice of the pre-quantized weight: _set_expert_offset_on_td
    # sets scalar_offset (x4 data) AND scalar_offset_scales (the companion scale tensor)
    # so each block reads its own expert's x4 rows + scales. Covers E in {2, 4} and
    # top_k in {1, 2, 4}. All shapes keep H != I_TP (see the E=1 method's note).
    _PREQUANT_ROUTED_PARAMS = [
        # H,    T,    E, B,   TOPK, I_TP
        pytest.param(1024, 1024, 2, 512, 1, 512, {"use_scale_packing": True}, marks=pytest.mark.fast, id="pq_e2_k1"),
        pytest.param(2048, 1024, 2, 128, 2, 256, {"use_scale_packing": True}, id="pq_e2_k2"),
        pytest.param(1024, 1024, 4, 512, 1, 128, {"use_scale_packing": True}, id="pq_e4_k1"),
        pytest.param(2048, 1024, 4, 256, 2, 384, {"use_scale_packing": True}, id="pq_e4_k2"),
        pytest.param(1024, 1024, 4, 512, 4, 256, {"use_scale_packing": True}, id="pq_e4_k4"),
        # Second E=2 shape (smaller I_TP). Packed scales throughout (see the E=1 note
        # on unpacked pre-quantized scales being a follow-up).
        pytest.param(1024, 1024, 2, 512, 1, 256, {"use_scale_packing": True}, id="pq_e2_small"),
        # Non-512 H on the routed path: the gate half's per-expert K-window (H/4 x4 rows) ends
        # mid-load-tile (H=768 = 512+256), exercising the per-half effective_f_dim clamp that
        # replaced the tile-aligned-H guard. gate/up split still via rhs_k_offset on the shared
        # expert-folded weight. (Real routed H is 512-aligned; this is edge-case coverage.)
        pytest.param(768, 1024, 2, 512, 1, 512, {"use_scale_packing": True}, id="pq_e2_h768"),
        pytest.param(768, 1024, 4, 256, 2, 384, {"use_scale_packing": True}, id="pq_e4_h768"),
    ]

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _PREQUANT_ROUTED_PARAMS,
    )
    def test_moe_mxfp8_fwd_prequantized_routed(
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
            prequantize_weights=True,
            **feature_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: pre-quantized weights loaded via the 1x32 PE-swizzle path.
    # ------------------------------------------------------------------
    # Same deployment story as the WRAPX pre-quantized test, but the weights are
    # pre-quantized with the 1x32 (contiguous-K) swizzle and the activation is loaded
    # via use_1x32_pe_swizzle, so both operands share the 1x32 K permutation. 1x32
    # requires single_expert_dense (E=1). Non-512 dims work on both axes: down K (I_TP) via
    # the flexible-K loader, and gate/up H via the quantized K-slice boundary clamp (e.g.
    # H=768). All shapes keep H != I_TP.
    _PREQUANT_1X32_PARAMS = [
        # H,    T,    E, B,    TOPK, I_TP
        pytest.param(512, 1024, 1, 512, 1, 1024, {}, marks=pytest.mark.fast, id="pq1x32_multi_ktile"),
        pytest.param(1024, 2048, 1, 512, 1, 512, {}, id="pq1x32_b512"),
        pytest.param(512, 1024, 1, 512, 1, 768, {}, id="pq1x32_itp768"),
        pytest.param(768, 1024, 1, 512, 1, 512, {}, id="pq1x32_h768"),
        pytest.param(1024, 1536, 1, 512, 1, 512, {}, id="pq1x32_N3_split"),
        # Qwen3-235B per-expert shapes, TP1 (I_TP=1536) and TP2 (I_TP=768), N=2.
        pytest.param(4096, 4096, 1, 2048, 1, 1536, {}, id="pq1x32_qwen3_tp1"),
        pytest.param(4096, 4096, 1, 2048, 1, 768, {}, id="pq1x32_qwen3_tp2"),
    ]

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, feature_kwargs",
        _PREQUANT_1X32_PARAMS,
    )
    def test_moe_mxfp8_fwd_prequantized_1x32(
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
            single_expert_dense=True,
            prequantize_weights=True,
            use_scale_packing=True,
            use_1x32_pe_swizzle=True,
            **feature_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: pre-quantized auto-detect with the kernel's own defaults (no config)
    # ------------------------------------------------------------------
    # The minimal deployment call: pass the x4 weights + scales and omit config +
    # block_size entirely. Passing scales auto-selects the pre-quantized path, and the
    # default config (MXFP8MOEFwdConfig()) resolves to the single_expert_dense 1x32
    # PE-swizzle load with scale packing on -- so no weight TD / load-mode flag is needed.
    # The offline weights are 1x32-swizzled + packed to match that default.
    @pytest.mark.fast
    def test_moe_mxfp8_fwd_prequantized_defaults_omit_config(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
    ):
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=256,
            tokens=8192,
            expert=1,
            block_size=4096,
            top_k=1,
            intermediate=512,
            single_expert_dense=True,
            prequantize_weights=True,
            use_scale_packing=True,
            use_1x32_pe_swizzle=True,
            omit_config=True,
            omit_block_size=True,
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
    # alone (no checkpoint) against the save layout. Everything else
    # (single_expert_dense, scale_packing, spill_reload, fast_dma_transpose) is held
    # fixed so the only variable is the per-block checkpoint store work.
    #   - ckpt_none: skip the checkpoint -> apples-to-apples vs the bf16 baseline
    #     (which emits none); the kernel spends no time on the store/alloc.
    #   - ckpt_direct: save gate/up, DIRECT (token-major [.., B, I_TP]) -> one plain
    #     DMA per block, no PE transpose.
    #   - ckpt_transposed: save gate/up, TRANSPOSED (I_TP-major [.., I_TP, B]) -> the
    #     PE block-transpose store, so this measures the transpose cost vs ckpt_direct.
    _QWEN3_CHECKPOINT_CONFIGS = [
        pytest.param({"save_gate_up_proj_act": False}, id="ckpt_none"),
        pytest.param({"gate_up_proj_act_layout": CheckpointLayout.DIRECT}, id="ckpt_direct"),
        pytest.param({"gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED}, id="ckpt_transposed"),
    ]

    # ------------------------------------------------------------------
    # Test: Qwen3-235B TP2/TP1 per-expert forward shapes
    # ------------------------------------------------------------------

    @pytest_parametrize(PARAM_NAMES, qwen3_235B_shapes, abbrevs=_ABBREVS)
    @pytest.mark.parametrize("checkpoint_kwargs", _QWEN3_CHECKPOINT_CONFIGS)
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
        checkpoint_kwargs: dict,
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
            single_expert_dense=True,
            use_scale_packing=True,
            spill_reload=True,
            # Load the unswizzled-BF16 operands via the fast direct-4D DGT path
            # (valid here because single_expert_dense carries no per-expert offset).
            fast_dma_transpose=True,
            **checkpoint_kwargs,
        )

    # ------------------------------------------------------------------
    # Test: Qwen3-235B TP1 shapes with the 1x32 PE-swizzle load mode
    # ------------------------------------------------------------------

    # Qwen3-235B TP1 (I_TP = 1536, H = 4096) deployment shapes: block_size in
    # {1024, 2048} with tokens = 2 * block_size gives N == 2 (both LNC2 cores busy).
    # block_size = 4096 is deferred until a separate large-block reshape limit is
    # confirmed clear on HW.
    qwen3_235B_tp1_1x32_shapes = [
        pytest.param(
            4096,
            2 * block_size,
            1,
            block_size,
            1,
            1536,
            marks=(pytest.mark.fast if block_size == 1024 else []),
            id=f"tp1_B{block_size}_N2_1x32",
        )
        for block_size in (1024, 2048)
    ]

    @pytest_parametrize(PARAM_NAMES, qwen3_235B_tp1_1x32_shapes, abbrevs=_ABBREVS)
    def test_moe_mxfp8_fwd_qwen3_235B_1x32_pe_swizzle(
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
        # Deployment-shape analog of test_moe_mxfp8_fwd_qwen3_235B_shapes with the load
        # mode swapped: 1x32 PE swizzle instead of fast_dma_transpose (DGT). Same
        # single_expert_dense + scale-packing + spill_reload configuration otherwise.
        self._run_test(
            test_manager=test_manager,
            platform_target=platform_target,
            hidden=hidden,
            tokens=tokens,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            intermediate=intermediate,
            single_expert_dense=True,
            use_scale_packing=True,
            spill_reload=True,
            use_1x32_pe_swizzle=True,
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
            single_expert_dense=True,
            use_scale_packing=True,
            spill_reload=True,
            reuse_spilled_weights=reuse_spilled_weights,
            fast_dma_transpose=True,
            # Match the benchmark checkpoint policy: save the gate/up checkpoint in the
            # DIRECT layout (plain DMA, no PE transpose) so the FFN cost dominates the
            # measurement. Both are the defaults, so no override is needed.
        )

    # ------------------------------------------------------------------
    # Test: DIRECT (non-transposed) checkpoint store layout
    # ------------------------------------------------------------------

    # Cover both gate/up checkpoint store layouts (DIRECT token-major, TRANSPOSED
    # I_TP-major) against the golden, which mirrors whichever layout the activation TD
    # orientation selects, plus the save-off case (no checkpoint emitted at all).
    _CHECKPOINT_LAYOUT_CONFIGS = [
        pytest.param({"gate_up_proj_act_layout": CheckpointLayout.DIRECT}, id="direct"),
        pytest.param({"gate_up_proj_act_layout": CheckpointLayout.TRANSPOSED}, id="transposed"),
        pytest.param({"save_gate_up_proj_act": False}, id="no_gate_up"),
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize("checkpoint_kwargs", _CHECKPOINT_LAYOUT_CONFIGS)
    def test_moe_mxfp8_fwd_checkpoint_layout(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        checkpoint_kwargs: dict,
    ):
        # Small single_expert_dense shape; validates the selected store layout against the
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
            single_expert_dense=True,
            **checkpoint_kwargs,
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

    # reuse_spilled_weights requires single_expert_dense (E=1), which the E>1
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
            single_expert_dense=True,
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
