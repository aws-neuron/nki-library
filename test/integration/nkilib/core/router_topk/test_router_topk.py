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

from test.integration.nkilib.core.router_topk.router_topk_torch import router_topk_torch_ref
from test.integration.nkilib.core.router_topk.test_router_topk_common import router_topk_tensor_gen
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import neuronxcc.nki.typing as nt
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
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

tensor_generator = gaussian_tensor_generator()


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
    Wrapper function for router_topk_kernel to handle SBUF input loading.

    Conditionally loads x tensor into SBUF if x_input_in_sbuf is True, then calls
    the main router_topk_kernel function.

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
        Output from router_topk_kernel
    """
    if x_input_in_sbuf:
        x_input_to_kernel = router_topk_input_x_load(x, hbm_layout=XHBMLayout_T_H__1, sb_layout=x_sb_layout)
    else:
        x_input_to_kernel = x

    # When output_in_sbuf=True, allocate SBUF tensors for outputs
    if output_in_sbuf:
        T, E = expert_affinities.shape
        expert_index_to_kernel = nl.ndarray((T, k), dtype=nl.uint32, buffer=nl.sbuf)
        expert_affinities_to_kernel = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    else:
        expert_index_to_kernel = expert_index
        expert_affinities_to_kernel = expert_affinities

    router_topk(
        x=x_input_to_kernel,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities_to_kernel,
        expert_index=expert_index_to_kernel,
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
        expert_affin_in_sb=output_in_sbuf,
    )

    # Copy SBUF results to HBM for validation
    if output_in_sbuf:
        nisa.dma_copy(src=expert_index_to_kernel, dst=expert_index)
        nisa.dma_copy(src=expert_affinities_to_kernel, dst=expert_affinities)

    return [router_logits, expert_index, expert_affinities]


def _generate_test_cases():
    """Generate all test case permutations by combining base configs with layout options."""
    # fmt: off
    # Base test grid: T, H, E, k, act_fn, has_bias, router_pre_norm, norm_topk_prob, use_column_tiling, use_indirect_dma_scatter, use_PE_broadcast_w_bias, shard_on_tokens, output_in_sbuf
    base_configs = [
        # Basic sweep of T
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (16,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (64,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (128,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        # No bias
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, False, False),
        # PE broadcast bias
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, True,  False, False),
        # E=256
        (4,    3072, 256, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        # High-batch
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (1024, 3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        # High-batch not divisible by 128
        (320,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        # LNC shard + not divisible by 128
        (48,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (80,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (96,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (160,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (192,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (320,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, True,  False),
        # Column tiling
        (16,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, True,  False, False, False, False),
        # Indirect DMA scatter
        (4,    3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, True,  False, False, False),
        (640,  3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, True,  False, False, False),
        # Sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,  False, False, False, True,  False, False, False),
        # router_pre_norm + softmax
        (8,    8192, 128, 8, RouterActFnType.SOFTMAX, True,  True,  False, False, True,  False, False, False),
        (640,  8192, 128, 8, RouterActFnType.SOFTMAX, True,  True,  False, False, True,  False, False, False),
        (1024, 8192, 128, 8, RouterActFnType.SOFTMAX, True,  True,  False, False, True,  False, False, False),
        # router_pre_norm + sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,  True,  False, False, True,  False, False, False),
        # router_pre_norm + norm_topk_prob + softmax
        (8,    8192, 128, 8, RouterActFnType.SOFTMAX, True,  True,  True,  False, True,  False, False, False),
        # router_pre_norm + norm_topk_prob + sigmoid
        (8,    8192, 128, 8, RouterActFnType.SIGMOID, True,  True,  True,  False, True,  False, False, False),
        # router_pre_norm with one-hot scatter (use_indirect_dma_scatter=False)
        (8,    3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  False, False, False, False, False, False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  False, False, False, False, False, False),
        (64,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  False, False, False, False, False, False),
        # router_pre_norm + norm_topk_prob with one-hot scatter
        (8,    3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  True,  False, False, False, False, False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  True,  False, False, False, False, False),
        # Mimic moe_token_gen megakernel test (k=1, no bias)
        (4,    3072, 128, 1, RouterActFnType.SOFTMAX, False, False, False, False, False, False, False, False),
        # Model-based configs: gptoss_120b
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  False, False, False, False, False, False, False),
        (32,   3072, 128, 4, RouterActFnType.SOFTMAX, True,  True,  False, False, True,  False, False, False),
        # qwen3_235b_a22b
        (2,    4096, 128, 8, RouterActFnType.SOFTMAX, False, False, False, False, False, False, False, False),
        (2,    4096, 128, 8, RouterActFnType.SOFTMAX, False, True,  False, False, True,  False, False, False),
        # llama4_maverick
        (2,    5120, 128, 1, RouterActFnType.SOFTMAX, False, False, False, False, False, False, False, False),
        (2,    5120, 128, 1, RouterActFnType.SOFTMAX, False, True,  False, False, True,  False, False, False),
        # llama4_scout
        (2,    5120, 16,  1, RouterActFnType.SOFTMAX, False, False, False, False, False, False, False, False),
        (2,    5120, 16,  1, RouterActFnType.SOFTMAX, False, True,  False, False, True,  False, False, False),
        # shard_on_tokens with output_in_sbuf (T<=128 required)
        (2,    512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (4,    512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (5,    512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (8,    512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (32,   512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (64,   512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
        (128,  512,  128, 4, RouterActFnType.SOFTMAX, False, False, False, False, False, False, True,  True),
    ]
    # fmt: on

    # Layout permutations
    x_input_in_sb_options = [True, False]
    x_hbm_layout_options = [XHBMLayout_H_T__0, XHBMLayout_T_H__1]
    x_sb_layout_options = [XSBLayout_tp102__0, XSBLayout_tp2013__1, XSBLayout_tp201__2]

    test_cases = []
    for base in base_configs:
        (
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
            output_in_sbuf,
        ) = base
        for x_input_in_sb in x_input_in_sb_options:
            for x_hbm_layout in x_hbm_layout_options:
                for x_sb_layout in x_sb_layout_options:
                    test_cases.append(
                        (
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
                            output_in_sbuf,
                            x_input_in_sb,
                            x_hbm_layout,
                            x_sb_layout,
                        )
                    )
    return test_cases


# Parameter names for pytest.mark.parametrize
ROUTER_TOPK_PARAMS = (
    "T, H, E, k, act_fn, has_bias, router_pre_norm, norm_topk_prob, "
    "use_column_tiling, use_indirect_dma_scatter, use_PE_broadcast_w_bias, "
    "shard_on_tokens, output_in_sbuf, x_input_in_sb, x_hbm_layout, x_sb_layout"
)

ROUTER_TOPK_TEST_CASES = _generate_test_cases()


@pytest_test_metadata(
    name="Router Top-K",
    pytest_marks=["router_topk", "moe"],
)
@final
class TestRouterTopkKernel:
    def run_router_topk_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
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
        """
        Execute a router top-K kernel test with the given parameters.

        Args:
            test_manager: Test orchestration manager
            compiler_args: Compiler configuration
            T: Number of tokens
            H: Hidden dimension
            E: Number of experts
            k: Top-K value
            act_fn: Activation function type
            has_bias: Whether to use bias
            router_pre_norm: Apply activation before top-K
            norm_topk_prob: Normalize top-K probabilities
            use_column_tiling: Enable PE array column tiling
            use_indirect_dma_scatter: Use indirect DMA for scatter
            use_PE_broadcast_w_bias: Use tensor engine for bias broadcast
            shard_on_tokens: Enable LNC sharding on tokens
            output_in_sbuf: Output in SBUF
            x_input_in_sb: Input x in SBUF
            x_hbm_layout: Layout of x in HBM
            x_sb_layout: Layout of x in SBUF
        """
        dtype = nl.bfloat16
        is_negative_test_case = False

        # Skip sigmoid + use_indirect_dma_scatter = False
        if act_fn == RouterActFnType.SIGMOID and not use_indirect_dma_scatter:
            is_negative_test_case = True

        T_local = T // 2 if shard_on_tokens else T
        if (T_local > 128 and T_local % 128 != 0) and use_indirect_dma_scatter:
            pytest.skip(f"Skipping test, not supported {T=} {use_indirect_dma_scatter=}")

        if (not x_input_in_sb) and (x_sb_layout != XSBLayout_tp102__0):
            pytest.skip("if 'x' is in HBM, x_sb_layout doesn't matter, so ensure only one combination is run.")
        if x_input_in_sb and (x_hbm_layout != XHBMLayout_H_T__0):
            pytest.skip("if 'x' is in SB, x_hbm_layout doesn't matter, so run only one combo")

        if output_in_sbuf and T > 128:
            pytest.skip(f"output_in_sbuf requires T <= 128, got {T=}")

        # Determine x_th_layout
        x_th_layout = x_input_in_sb or x_hbm_layout == XHBMLayout_T_H__1

        # Generate kernel input tensors
        x_shape = (T, H) if x_th_layout else (H, T)
        w_shape = (H, E)
        router_logits_shape = (T, E)

        kernel_input_args = {}
        kernel_input_args["x"] = router_topk_tensor_gen(name="x", shape=x_shape, dtype=dtype)
        kernel_input_args["w"] = router_topk_tensor_gen(name="w", shape=w_shape, dtype=dtype)
        kernel_input_args["w_bias"] = (
            router_topk_tensor_gen(name="w_bias", shape=(1, E), dtype=dtype) if has_bias else None
        )

        kernel_input_args["act_fn"] = act_fn
        kernel_input_args["k"] = k
        kernel_input_args["x_hbm_layout"] = x_hbm_layout
        kernel_input_args["x_sb_layout"] = x_sb_layout
        kernel_input_args["output_in_sbuf"] = output_in_sbuf
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

        kernel_input_args["router_logits.must_alias_input"] = tensor_generator(
            shape=router_logits_shape, dtype=dtype, name="router_logits"
        )
        expert_affinities_dtype = nl.float32 if output_in_sbuf else dtype
        kernel_input_args["expert_affinities.must_alias_input"] = tensor_generator(
            shape=(T, E), dtype=expert_affinities_dtype, name="expert_affinities"
        )
        kernel_input_args["expert_index.must_alias_input"] = tensor_generator(
            shape=(T, k), dtype=nl.uint32, name="expert_index"
        )

        output_tensors = {
            "router_logits": np.zeros(router_logits_shape, dtype=dtype),
            "expert_index": np.zeros((T, k), dtype=nl.uint32),
            "expert_affinities": np.zeros((T, E), dtype=expert_affinities_dtype),
        }

        with assert_negative_test_case(is_negative_test_case):

            def create_lazy_golden():
                return router_topk_torch_ref(
                    x=kernel_input_args["x"],
                    w=kernel_input_args["w"],
                    w_bias=kernel_input_args["w_bias"],
                    router_logits=output_tensors["router_logits"],
                    expert_affinities=output_tensors["expert_affinities"],
                    expert_index=output_tensors["expert_index"],
                    act_fn=act_fn,
                    k=k,
                    x_hbm_layout=x_hbm_layout,
                    x_sb_layout=x_sb_layout,
                    output_in_sbuf=output_in_sbuf,
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
                    expert_affin_in_sb=output_in_sbuf,
                )

            test_manager.execute(
                KernelArgs(
                    kernel_func=kernel_wrapper,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input_args,
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
        """
        Main test method for router top-K kernel.

        Tests various T, H, E, k combinations with different activation functions,
        layouts, and optimization modes.
        """
        compiler_args = CompilerArgs()
        self.run_router_topk_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
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
