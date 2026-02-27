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

import random
from test.integration.nkilib.core.router_topk.router_topk_torch import router_topk_torch_ref
from test.integration.nkilib.core.router_topk.test_router_topk_common import router_topk_tensor_gen
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

import neuronxcc.nki.typing as nt
import nki.language as nl
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from nkilib_src.nkilib.core.router_topk.router_topk import (
    XHBMLayout_H_T__0,
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    XSBLayout_tp201__2,
    XSBLayout_tp2013__1,
    router_topk,
    router_topk_input_x_load,
)
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType


def kernel_wrapper(
    x: nl.ndarray,
    w: nl.ndarray,
    w_bias: nl.ndarray,
    router_logits: nt.mutable_tensor,
    expert_affinities: nt.mutable_tensor,
    expert_index: nt.mutable_tensor,
    act_fn: RouterActFnType,
    k: int,
    x_hbm_layout: XHBMLayout_H_T__0,
    x_sb_layout: XSBLayout_tp102__0,
    output_in_sbuf: bool = False,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    use_column_tiling: bool = False,
    use_indirect_dma_scatter: bool = False,
    return_eager_affi: bool = False,
    use_PE_broadcast_w_bias: bool = False,
    shard_on_tokens: bool = False,
    skip_store_expert_index: bool = False,
    skip_store_router_logits: bool = False,
    x_input_in_sbuf: bool = False,
):
    """
    Wrapper function for router_topk kernel to handle SBUF input loading.

    Conditionally loads x tensor into SBUF if x_input_in_sbuf is True, then calls
    the main router_topk kernel function.

    Args:
        x (nl.ndarray): Input tensor
        w (nl.ndarray): Weight tensor
        w_bias (nl.ndarray): Optional bias tensor
        router_logits (nt.mutable_tensor): Output router logits
        expert_affinities (nt.mutable_tensor): Output expert affinities
        expert_index (nt.mutable_tensor): Output expert indices
        act_fn (RouterActFnType): Activation function
        k (int): Number of top experts to select
        x_hbm_layout: Layout of x in HBM
        x_sb_layout: Layout of x in SBUF
        output_in_sbuf (bool): If True, outputs are in SBUF
        router_pre_norm (bool): If True, apply activation before top-K
        norm_topk_prob (bool): If True, normalize top-K probabilities
        use_column_tiling (bool): Enable PE array column tiling
        use_indirect_dma_scatter (bool): Use indirect DMA for scatter
        return_eager_affi (bool): If True, return top-K affinities
        use_PE_broadcast_w_bias (bool): Use tensor engine for bias broadcast
        shard_on_tokens (bool): Enable LNC sharding on tokens
        skip_store_expert_index (bool): Skip storing expert indices
        skip_store_router_logits (bool): Skip storing router logits
        x_input_in_sbuf (bool): If True, load x into SBUF before kernel call

    Returns:
        Output from router_topk kernel
    """
    if x_input_in_sbuf:
        x_input_to_kernel = router_topk_input_x_load(x, hbm_layout=XHBMLayout_T_H__1, sb_layout=x_sb_layout)
    else:
        x_input_to_kernel = x

    return router_topk(
        x=x_input_to_kernel,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
        output_in_sbuf=output_in_sbuf,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        use_column_tiling=use_column_tiling,
        use_indirect_dma_scatter=use_indirect_dma_scatter,
        return_eager_affi=return_eager_affi,
        use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
        shard_on_tokens=shard_on_tokens,
        skip_store_expert_index=skip_store_expert_index,
        skip_store_router_logits=skip_store_router_logits,
        x_input_in_sbuf=x_input_in_sbuf,
    )


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

    # Invalid: T_local > 128 and not divisible by 128 with indirect DMA scatter
    if (T_local > 128 and T_local % 128 != 0) and use_indirect_dma_scatter:
        return FilterResult.INVALID

    # Redundant: If x is in HBM, x_sb_layout doesn't matter - only run one combination
    if (not x_input_in_sb) and (x_sb_layout != XSBLayout_tp102__0):
        return FilterResult.REDUNDANT

    # Redundant: If x is in SBUF, x_hbm_layout doesn't matter - only run one combination
    if x_input_in_sb and (x_hbm_layout != XHBMLayout_H_T__0):
        return FilterResult.REDUNDANT

    return FilterResult.VALID


def generate_kernel_inputs(
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
    Generate inputs for the router_topk kernel.

    Creates all necessary input tensors and configuration parameters for
    running the router_topk kernel test.

    Args:
        T (int): Number of tokens
        H (int): Hidden dimension size
        E (int): Number of experts
        k (int): Number of top experts to select
        act_fn (RouterActFnType): Activation function
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
        tuple: (kernel_input_args dict, x_th_layout bool, dtype)
    """
    dtype = nl.bfloat16
    tensor_generator = gaussian_tensor_generator()

    # Determine x_th_layout: True if x is [T, H], False if [H, T]
    x_th_layout = x_input_in_sb or x_hbm_layout == XHBMLayout_T_H__1

    # Generate tensor shapes
    x_shape = (T, H) if x_th_layout else (H, T)
    w_shape = (H, E)
    router_logits_shape = (T, E)

    kernel_input_args = {}

    # Generate input tensors
    kernel_input_args["x"] = router_topk_tensor_gen(name="x", shape=x_shape, dtype=dtype)
    kernel_input_args["w"] = router_topk_tensor_gen(name="w", shape=w_shape, dtype=dtype)

    if has_bias:
        kernel_input_args["w_bias"] = router_topk_tensor_gen(name="w_bias", shape=(1, E), dtype=dtype)
    else:
        kernel_input_args["w_bias"] = None

    # Configuration parameters
    kernel_input_args["act_fn"] = act_fn
    kernel_input_args["k"] = k
    kernel_input_args["x_hbm_layout"] = x_hbm_layout
    kernel_input_args["x_sb_layout"] = x_sb_layout
    kernel_input_args["output_in_sbuf"] = False
    kernel_input_args["router_pre_norm"] = router_pre_norm
    kernel_input_args["norm_topk_prob"] = norm_topk_prob
    kernel_input_args["use_column_tiling"] = use_column_tiling
    kernel_input_args["use_indirect_dma_scatter"] = use_indirect_dma_scatter
    kernel_input_args["return_eager_affi"] = False
    kernel_input_args["use_PE_broadcast_w_bias"] = use_PE_broadcast_w_bias
    kernel_input_args["shard_on_tokens"] = shard_on_tokens
    kernel_input_args["skip_store_expert_index"] = False
    kernel_input_args["skip_store_router_logits"] = False
    kernel_input_args["x_input_in_sbuf"] = x_input_in_sb

    # Output tensors (must_alias_input pattern)
    kernel_input_args["router_logits.must_alias_input"] = tensor_generator(
        shape=router_logits_shape, dtype=dtype, name='router_logits'
    )
    kernel_input_args["expert_affinities.must_alias_input"] = tensor_generator(
        shape=(T, E), dtype=dtype, name='expert_affinities'
    )
    kernel_input_args["expert_index.must_alias_input"] = tensor_generator(
        shape=(T, k), dtype=nl.uint32, name='expert_index'
    )

    return kernel_input_args, x_th_layout, dtype


@pytest_test_metadata(
    name="Router Top-K Sweep",
    pytest_marks=["router_topk", "moe", "sweep"],
)
# @IGNORE_FAST
class TestRouterTopkSweep:
    """
    Sweep test class for router_topk kernel.

    Uses pytest coverage_parametrize to generate pairwise test combinations
    covering dimension parameters, activation functions, layouts, and
    optimization flags.
    """

    @pytest.mark.coverage_parametrize(
        # T has wide valid range - disable boundary tests for it
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
        Sweep test for router_topk kernel with pairwise parameter coverage.

        Tests the router_topk kernel across various parameter combinations using
        coverage_parametrize for efficient pairwise testing.

        Args:
            test_manager (Orchestrator): Test orchestration manager
            T (int): Number of tokens
            H (int): Hidden dimension size
            E (int): Number of experts
            k (int): Number of top experts to select
            act_fn (RouterActFnType): Activation function
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
        """
        kernel_input, x_th_layout, dtype = generate_kernel_inputs(
            T=T,
            H=H,
            E=E,
            k=k,
            act_fn=act_fn,
            has_bias=has_bias,
            router_pre_norm=router_pre_norm,
            norm_topk_prob=norm_topk_prob,
            x_input_in_sb=x_input_in_sb,
            x_hbm_layout=x_hbm_layout,
            x_sb_layout=x_sb_layout,
            use_column_tiling=use_column_tiling,
            use_indirect_dma_scatter=use_indirect_dma_scatter,
            use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
            shard_on_tokens=shard_on_tokens,
        )

        compiler_args = CompilerArgs()

        # Create output tensors for golden function
        output_tensors = {
            "router_logits": np.zeros((T, E), dtype=dtype),
            "expert_index": np.zeros((T, k), dtype=nl.uint32),
            "expert_affinities": np.zeros((T, E), dtype=dtype),
        }

        # Create closure to capture all variables needed for lazy golden computation
        def create_lazy_golden():
            return router_topk_torch_ref(
                x=kernel_input["x"],
                w=kernel_input["w"],
                w_bias=kernel_input["w_bias"],
                router_logits=output_tensors["router_logits"],
                expert_affinities=output_tensors["expert_affinities"],
                expert_index=output_tensors["expert_index"],
                act_fn=act_fn,
                k=k,
                x_hbm_layout=x_hbm_layout,
                x_sb_layout=x_sb_layout,
                output_in_sbuf=False,
                router_pre_norm=router_pre_norm,
                norm_topk_prob=norm_topk_prob,
                use_column_tiling=use_column_tiling,
                use_indirect_dma_scatter=use_indirect_dma_scatter,
                return_eager_affi=False,
                use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
                shard_on_tokens=shard_on_tokens,
                skip_store_expert_index=False,
                skip_store_router_logits=False,
                x_input_in_sbuf=x_input_in_sb,
                expert_affin_in_sb=False,
            )

        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=kernel_wrapper,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=create_lazy_golden,
                            output_ndarray=output_tensors,
                        ),
                        absolute_accuracy=1e-5,
                        relative_accuracy=2e-2,
                    ),
                )
            )
