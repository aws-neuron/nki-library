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

"""Integration tests for the router top-K kernel. Tests various configurations including different tensor layouts, activation functions, and optimization modes."""

import itertools
import random
from test.integration.nkilib.core.router_topk.test_router_topk_common import (
    generate_router_topk_inputs,
    router_topk_kernel_wrapper,
    router_topk_torch_wrapper,
)
from test.utils.common_dataclasses import CompilerArgs
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import pytest
from nkilib_src.nkilib.core.router_topk.router_topk import (
    XHBMLayout_H_T__0,
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    XSBLayout_tp201__2,
    XSBLayout_tp2013__1,
)
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType


def _generate_test_cases():
    """Generate all test case permutations by combining base configs with layout options."""
    # fmt: off
    base_config_params = \
        "T,    H,    E,   k, act_fn,                  has_bias, router_pre_norm, norm_topk_prob, use_column_tiling, use_indirect_dma_scatter, use_PE_broadcast_w_bias, shard_on_tokens, output_in_sbuf"
    base_configs = [
        # Basic sweep of T
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (16,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (64,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (128,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        # No bias
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   False,           False),
        # PE broadcast bias
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    True,                    False,           False),
        # E=256
        (4,    3072, 256, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        # High-batch
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (1024, 3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        # High-batch not divisible by 128
        (320,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        # LNC shard + not divisible by 128
        (48,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (80,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (96,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (160,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (192,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (320,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   True,            False),
        # Column tiling
        (16,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          True,              False,                    False,                   False,           False),
        # Indirect DMA scatter
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             True,                     False,                   False,           False),
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             True,                     False,                   False,           False),
        # Sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,     False,           False,          False,             True,                     False,                   False,           False),
        # router_pre_norm + softmax
        (8,    8192, 128, 8, RouterActFnType.SOFTMAX, True,     True,            False,          False,             True,                     False,                   False,           False),
        (640,  8192, 128, 8, RouterActFnType.SOFTMAX, True,     True,            False,          False,             True,                     False,                   False,           False),
        (1024, 8192, 128, 8, RouterActFnType.SOFTMAX, True,     True,            False,          False,             True,                     False,                   False,           False),
        # router_pre_norm + sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,     True,            False,          False,             True,                     False,                   False,           False),
        # router_pre_norm + norm_topk_prob + softmax
        (8,    8192, 128, 8, RouterActFnType.SOFTMAX, True,     True,            True,           False,             True,                     False,                   False,           False),
        # router_pre_norm + norm_topk_prob + sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,     True,            True,           False,             True,                     False,                   False,           False),
        # router_pre_norm with one-hot scatter (use_indirect_dma_scatter=False)
        (8,    3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            False,          False,             False,                    False,                   False,           False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            False,          False,             False,                    False,                   False,           False),
        (64,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            False,          False,             False,                    False,                   False,           False),
        # router_pre_norm + norm_topk_prob with one-hot scatter
        (8,    3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            True,           False,             False,                    False,                   False,           False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            True,           False,             False,                    False,                   False,           False),
        # Mimic moe_token_gen megakernel test (k=1, no bias)
        (4,    3072, 128, 1, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   False,           False),
        # Model-based configs: gptoss_120b
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     False,           False,          False,             False,                    False,                   False,           False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,     True,            False,          False,             True,                     False,                   False,           False),
        # qwen3_235b_a22b
        (2,    4096, 128, 8, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   False,           False),
        (2,    4096, 128, 8, RouterActFnType.SOFTMAX, False,    True,            False,          False,             True,                     False,                   False,           False),
        # llama4_maverick
        (2,    5120, 128, 1, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   False,           False),
        (2,    5120, 128, 1, RouterActFnType.SOFTMAX, False,    True,            False,          False,             True,                     False,                   False,           False),
        # llama4_scout
        (2,    5120, 16,  1, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   False,           False),
        (2,    5120, 16,  1, RouterActFnType.SOFTMAX, False,    True,            False,          False,             True,                     False,                   False,           False),
        # shard_on_tokens with output_in_sbuf (T<=128 required)
        (2,    512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (4,    512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (5,    512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (8,    512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (32,   512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (64,   512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
        (128,  512,  128, 4, RouterActFnType.SOFTMAX, False,    False,           False,          False,             False,                    False,                   True,            True),
    ]
    # fmt: on

    # Layout permutations
    x_input_in_sb_options = [True, False]
    x_hbm_layout_options = [XHBMLayout_H_T__0, XHBMLayout_T_H__1]
    x_sb_layout_options = [XSBLayout_tp102__0, XSBLayout_tp2013__1, XSBLayout_tp201__2]

    # Use itertools.product to avoid manual unpacking/repacking errors
    test_cases = [
        base + (x_input_in_sb, x_hbm_layout, x_sb_layout)
        for base in base_configs
        for x_input_in_sb, x_hbm_layout, x_sb_layout in itertools.product(
            x_input_in_sb_options, x_hbm_layout_options, x_sb_layout_options
        )
    ]

    params = ", ".join(p.strip() for p in base_config_params.split(",")) + ", x_input_in_sb, x_hbm_layout, x_sb_layout"
    return params, test_cases


ROUTER_TOPK_PARAMS, ROUTER_TOPK_TEST_CASES = _generate_test_cases()


def filter_illegal_combinations(
    T,
    H,
    E,
    k,
    act_fn,
    has_bias,
    router_pre_norm,
    norm_topk_prob,
    x_input_in_sb,
    x_hbm_layout,
    x_sb_layout,
    use_column_tiling,
    use_indirect_dma_scatter,
    use_PE_broadcast_w_bias,
    shard_on_tokens,
):
    """
    Filter out illegal parameter combinations for router_topk kernel.

    This function encodes the skip logic from the unit test to prevent running
    invalid configurations that would fail or produce undefined behavior.

    Uses default None values for better AllPairs optimization - allows partial
    combinations during test generation.

    Args:
        T (int): Number of tokens
        H (int): Hidden dimension size
        E (int): Number of experts
        k (int): Number of top experts to select
        act_fn (RouterActFnType): Activation function (SOFTMAX or SIGMOID)
        has_bias (bool): Whether to include bias
        router_pre_norm (bool): If True, apply activation before top-K
        norm_topk_prob (bool): If True, normalize top-K probabilities
        x_input_in_sb (bool): If True, x is in SBUF
        x_hbm_layout: Layout of x in HBM
        x_sb_layout: Layout of x in SBUF
        use_column_tiling (bool): Enable PE array column tiling
        use_indirect_dma_scatter (bool): Use indirect DMA for scatter
        use_PE_broadcast_w_bias (bool): Use tensor engine for bias broadcast
        shard_on_tokens (bool): Enable LNC sharding on tokens

    Returns:
        FilterResult: VALID, INVALID, or REDUNDANT
    """
    # Invalid: SIGMOID requires indirect DMA scatter
    if act_fn == RouterActFnType.SIGMOID and not use_indirect_dma_scatter:
        return FilterResult.INVALID

    # Redundant: norm_topk_prob requires router_pre_norm
    if norm_topk_prob and not router_pre_norm:
        return FilterResult.REDUNDANT

    T_local = T // 2 if shard_on_tokens else T
    if (T_local > 128 and T_local % 128 != 0) and use_indirect_dma_scatter:
        return FilterResult.INVALID
    if (not x_input_in_sb) and (x_sb_layout != XSBLayout_tp102__0):
        return FilterResult.REDUNDANT
    if x_input_in_sb and (x_hbm_layout != XHBMLayout_H_T__0):
        return FilterResult.REDUNDANT

    return FilterResult.VALID


@pytest_test_metadata(
    name="Router Top-K",
    pytest_marks=["router_topk", "moe"],
)
class TestRouterTopkKernel:
    """Test class for router_topk using UnitTestFramework."""

    @pytest.mark.fast
    @pytest.mark.parametrize(ROUTER_TOPK_PARAMS, ROUTER_TOPK_TEST_CASES)
    def test_router_topk(
        self,
        test_manager: Orchestrator,
        T: int,
        H: int,
        E: int,
        k: int,
        act_fn: RouterActFnType,
        has_bias: bool,
        router_pre_norm: bool,
        norm_topk_prob: bool,
        use_column_tiling: bool,
        use_indirect_dma_scatter: bool,
        use_PE_broadcast_w_bias: bool,
        shard_on_tokens: bool,
        output_in_sbuf: bool,
        x_input_in_sb: bool,
        x_hbm_layout,
        x_sb_layout,
    ):
        """Test router_topk using UnitTestFramework."""
        # Skip conditions
        T_local = T // 2 if shard_on_tokens else T
        if (T_local > 128 and T_local % 128 != 0) and use_indirect_dma_scatter:
            pytest.skip(f"Skipping test, not supported {T=} {use_indirect_dma_scatter=}")
        if (not x_input_in_sb) and (x_sb_layout != XSBLayout_tp102__0):
            pytest.skip("if 'x' is in HBM, x_sb_layout doesn't matter, so ensure only one combination is run.")
        if x_input_in_sb and (x_hbm_layout != XHBMLayout_H_T__0):
            pytest.skip("if 'x' is in SB, x_hbm_layout doesn't matter, so run only one combo")
        if output_in_sbuf and T > 128:
            pytest.skip(f"output_in_sbuf requires T <= 128, got {T=}")

        def input_generator(test_config):
            return generate_router_topk_inputs(
                T=T,
                H=H,
                E=E,
                k=k,
                act_fn=act_fn,
                has_bias=has_bias,
                router_pre_norm=router_pre_norm,
                norm_topk_prob=norm_topk_prob,
                use_column_tiling=use_column_tiling,
                use_indirect_dma_scatter=use_indirect_dma_scatter,
                use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
                shard_on_tokens=shard_on_tokens,
                output_in_sbuf=output_in_sbuf,
                x_input_in_sb=x_input_in_sb,
                x_hbm_layout=x_hbm_layout,
                x_sb_layout=x_sb_layout,
            )

        def output_tensors(kernel_input):
            return {
                "router_logits": kernel_input["router_logits.must_alias_input"].copy(),
                "expert_index": kernel_input["expert_index.must_alias_input"].copy(),
                "expert_affinities": kernel_input["expert_affinities.must_alias_input"].copy(),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=router_topk_kernel_wrapper,
            torch_ref=torch_ref_wrapper(router_topk_torch_wrapper),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            check_unused_params=True,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.coverage_parametrize(
        T=BoundedRange(random.sample(range(4, 1024), 20), boundary_values=[]),
        H=[3072, 8192],
        E=BoundedRange([128, 256], boundary_values=[513]),
        k=[1, 4, 8],
        act_fn=[RouterActFnType.SOFTMAX, RouterActFnType.SIGMOID],
        has_bias=[True, False],
        router_pre_norm=[True, False],
        norm_topk_prob=[True, False],
        x_input_in_sb=[True, False],
        x_hbm_layout=[XHBMLayout_H_T__0, XHBMLayout_T_H__1],
        x_sb_layout=[XSBLayout_tp102__0, XSBLayout_tp2013__1, XSBLayout_tp201__2],
        use_column_tiling=[True, False],
        use_indirect_dma_scatter=[True, False],
        use_PE_broadcast_w_bias=[True, False],
        shard_on_tokens=[True, False],
        filter=filter_illegal_combinations,
        coverage="pairs",
    )
    def test_router_topk_sweep(
        self,
        test_manager: Orchestrator,
        T,
        H,
        E,
        k,
        act_fn,
        has_bias,
        router_pre_norm,
        norm_topk_prob,
        x_input_in_sb,
        x_hbm_layout,
        x_sb_layout,
        use_column_tiling,
        use_indirect_dma_scatter,
        use_PE_broadcast_w_bias,
        shard_on_tokens,
        is_negative_test_case,
    ):
        """
        Sweep test for the router top-K kernel using coverage_parametrize.

        This test module provides comprehensive coverage of the router_topk kernel using
        pairwise parameter coverage. It tests various combinations of:
        - Dimension parameters (T, H, E, k)
        - Activation functions (SOFTMAX, SIGMOID)
        - Layout configurations (HBM and SBUF layouts)
        - Optimization flags (column tiling, indirect DMA scatter, etc.)

        The sweep test uses a filter function to skip invalid parameter combinations
        and validates results against a PyTorch golden reference implementation.
        """

        def input_generator(test_config):
            return generate_router_topk_inputs(
                T,
                H,
                E,
                k,
                act_fn,
                has_bias,
                router_pre_norm,
                norm_topk_prob,
                use_column_tiling,
                use_indirect_dma_scatter,
                use_PE_broadcast_w_bias,
                shard_on_tokens,
                False,
                x_input_in_sb,
                x_hbm_layout,
                x_sb_layout,
            )

        def output_tensors(kernel_input):
            return {
                "router_logits": kernel_input["router_logits.must_alias_input"].copy(),
                "expert_index": kernel_input["expert_index.must_alias_input"].copy(),
                "expert_affinities": kernel_input["expert_affinities.must_alias_input"].copy(),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=router_topk_kernel_wrapper,
            torch_ref=torch_ref_wrapper(router_topk_torch_wrapper),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            check_unused_params=True,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )
