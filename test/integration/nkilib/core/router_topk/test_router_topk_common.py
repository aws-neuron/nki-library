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

"""Test utilities for router top-K tests."""

import neuron_dtypes as dt
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from nkilib_src.nkilib.core.router_topk.router_topk import (
    XHBMLayout_T_H__1,
    router_topk,
    router_topk_input_x_load,
)
from nkilib_src.nkilib.core.router_topk.router_topk_torch import router_topk_torch_ref


def router_topk_tensor_gen(name: str, shape, dtype):
    """
    Generate test tensor with values from [-0.1, 0.1] for router top-K testing.

    Args:
        name (str): Tensor name (unused, for compatibility)
        shape: Shape of the tensor to generate
        dtype: Data type for the tensor

    Returns:
        numpy.ndarray: Generated tensor with uniform random values in [-0.1, 0.1]

    Notes:
        - Uses thread-safe Generator for reproducibility
        - Range chosen because x.T @ w may go into sigmoid activation
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    tensor = torch.empty(shape, dtype=torch.float32).uniform_(-0.1, 0.1, generator=generator)
    return dt.static_cast(tensor.numpy(), dtype)


def router_topk_kernel_wrapper(
    x,
    w,
    w_bias,
    router_logits,
    expert_affinities,
    expert_index,
    act_fn,
    k,
    x_hbm_layout,
    x_sb_layout,
    output_in_sbuf=False,
    router_pre_norm=True,
    norm_topk_prob=False,
    use_column_tiling=False,
    use_indirect_dma_scatter=False,
    return_eager_affi=False,
    use_PE_broadcast_w_bias=False,
    shard_on_tokens=False,
    skip_store_expert_index=False,
    skip_store_router_logits=False,
    x_input_in_sbuf=False,
):
    """Wrapper for router_topk that handles SBUF I/O."""
    if x_input_in_sbuf:
        x_input = router_topk_input_x_load(x, hbm_layout=XHBMLayout_T_H__1, sb_layout=x_sb_layout)
    else:
        x_input = x

    # When output_in_sbuf=True, allocate SBUF tensors for outputs
    if output_in_sbuf:
        T, E = expert_affinities.shape
        expert_index_to_kernel = nl.ndarray((T, k), dtype=nl.uint32, buffer=nl.sbuf)
        expert_affinities_to_kernel = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    else:
        expert_index_to_kernel = expert_index
        expert_affinities_to_kernel = expert_affinities

    router_topk(
        x=x_input,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities_to_kernel,
        expert_index=expert_index_to_kernel,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        use_column_tiling=use_column_tiling,
        use_indirect_dma_scatter=use_indirect_dma_scatter,
        return_eager_affi=return_eager_affi,
        use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
        shard_on_tokens=shard_on_tokens,
        skip_store_expert_index=skip_store_expert_index,
        skip_store_router_logits=skip_store_router_logits,
    )

    # Copy SBUF results to HBM for validation
    if output_in_sbuf:
        nisa.dma_copy(src=expert_index_to_kernel, dst=expert_index)
        nisa.dma_copy(src=expert_affinities_to_kernel, dst=expert_affinities)

    return [router_logits, expert_index, expert_affinities]


def router_topk_torch_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bias: torch.Tensor,
    router_logits: torch.Tensor,
    expert_affinities: torch.Tensor,
    expert_index: torch.Tensor,
    act_fn: "RouterActFnType",
    k: int,
    x_hbm_layout: int,
    x_sb_layout: int,
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
) -> list[torch.Tensor]:
    """Wrapper for router_topk_torch_ref matching router_topk_kernel_wrapper API."""
    return router_topk_torch_ref(
        x=x,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
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


def generate_router_topk_inputs(
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
):
    """Generate inputs for router_topk test."""
    dtype = nl.bfloat16
    expert_affinities_dtype = nl.float32 if output_in_sbuf else dtype

    x_th_layout = x_input_in_sb or x_hbm_layout == 1  # XHBMLayout_T_H__1
    x_shape = (T, H) if x_th_layout else (H, T)

    return {
        "x": router_topk_tensor_gen(name="x", shape=x_shape, dtype=dtype),
        "w": router_topk_tensor_gen(name="w", shape=(H, E), dtype=dtype),
        "w_bias": (router_topk_tensor_gen(name="w_bias", shape=(1, E), dtype=dtype) if has_bias else None),
        "router_logits.must_alias_input": np.zeros((T, E), dtype=dtype),
        "expert_affinities.must_alias_input": np.zeros((T, E), dtype=expert_affinities_dtype),
        "expert_index.must_alias_input": np.zeros((T, k), dtype=np.uint32),
        "act_fn": act_fn,
        "k": k,
        "x_hbm_layout": x_hbm_layout,
        "x_sb_layout": x_sb_layout,
        "router_pre_norm": router_pre_norm,
        "norm_topk_prob": norm_topk_prob,
        "use_column_tiling": use_column_tiling,
        "use_indirect_dma_scatter": use_indirect_dma_scatter,
        "use_PE_broadcast_w_bias": use_PE_broadcast_w_bias,
        "shard_on_tokens": shard_on_tokens,
        "output_in_sbuf": output_in_sbuf,
        "x_input_in_sbuf": x_input_in_sb,
    }
