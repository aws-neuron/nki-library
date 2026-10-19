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
    P_MAX,
    XHBMLayout_T_H__1,
    XSBLayout_tp2013__1,
    _hbm_tiled_store_view,
    router_topk,
    router_topk_input_x_load,
)
from nkilib_src.nkilib.core.router_topk.router_topk_torch import router_topk_torch_ref
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType

from test.utils.rng import NKITestsRNG

_rng = NKITestsRNG()


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
        - Range chosen because x.T @ w may go into sigmoid activation
    """
    tensor = torch.empty(shape, dtype=torch.float32)
    _rng.reset()
    _rng.tensor_uniform_(tensor, -0.1, 0.1)
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
    expert_affinities_eager_out=None,
):
    """Wrapper for router_topk that handles SBUF I/O."""
    # Infer x_input_in_sbuf: if x is (T, H) but x_hbm_layout says (H, T), x was intended for SBUF
    H = w.shape[0]
    x_input_in_sbuf = (x.shape[0] != H) and (x_hbm_layout == 0)
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

    kernel_outputs = router_topk(
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
        expert_affinities_eager_out=expert_affinities_eager_out,
    )

    # Copy SBUF results to HBM for validation. The non-sharded SBUF path rebinds
    # expert_affinities to a reshaped internal buffer, so read it from the kernel's return.
    if output_in_sbuf:
        scattered_expert_affinities = kernel_outputs[2]
        nisa.dma_copy(src=expert_index_to_kernel, dst=expert_index)
        nisa.dma_copy(src=scattered_expert_affinities, dst=expert_affinities)

    result = [router_logits, expert_index, expert_affinities]
    if expert_affinities_eager_out is not None:
        result.append(expert_affinities_eager_out)
    return result


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
    expert_affinities_eager_out=None,
) -> list[torch.Tensor]:
    """Wrapper for router_topk_torch_ref matching router_topk_kernel_wrapper API."""
    # Infer if x is in T×H layout from shape: w is always (H, E), so w.shape[0] == H.
    # If x.shape[0] != H, x must be (T, H) — override x_hbm_layout so the torch_ref
    # handles the transpose correctly.
    H = w.shape[0]
    x_in_th_layout = x.shape[0] != H
    result = router_topk_torch_ref(
        x=x,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=1 if x_in_th_layout else x_hbm_layout,
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

    # When a store is skipped, the kernel never writes that output, so its buffer stays
    # at its zero-initialized value. Mirror that in the golden so validation matches.
    if skip_store_router_logits:
        result["router_logits"] = torch.zeros_like(result["router_logits"])
    if skip_store_expert_index:
        result["expert_index"] = torch.zeros_like(result["expert_index"])

    # HBM-direct eager output: the dense [T, k] top-K affinities in top-K order. These are
    # exactly the scattered affinities gathered back at the top-K expert indices.
    if expert_affinities_eager_out is not None:
        eager = torch.gather(result["expert_affinities"], 1, result["expert_index"].long())
        result["expert_affinities_eager_out"] = eager

    return result


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
    }


# ---------------------------------------------------------------------------
# Dedicated wrappers for two branches the standard wrapper cannot express:
#   1) sharded SBUF expert_index with T>128 (needs a 3-D [128, num_t_tiles, k]
#      SBUF output tensor, not the 2-D [T, k] the standard wrapper builds), and
#   2) the layout-1 weight load with an ODD num_h_tiles (the "unbalanced" path),
#      which requires a layout-1 SBUF x that router_topk_input_x_load cannot
#      produce for odd num_h_tiles.
# ---------------------------------------------------------------------------


def router_topk_sharded_sbuf_index_kernel_wrapper(
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
):
    """Sharded run with expert_index in SBUF for T>128 (3-D tile-based store path)."""
    T = expert_index.shape[0]
    num_t_tiles_total = (T + P_MAX - 1) // P_MAX
    # SBUF expert_index must be [128, num_t_tiles, k] for the T>128 sharded path;
    # a flat [T, k] would exceed the 128-partition limit. expert_affinities stays in HBM.
    expert_index_sb = nl.ndarray((P_MAX, num_t_tiles_total, k), dtype=nl.uint32, buffer=nl.sbuf)
    router_topk(
        x=x,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index_sb,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
        router_pre_norm=False,
        shard_on_tokens=True,
    )
    # Fold the [128, num_t_tiles, k] SBUF result back into the [T, k] HBM output.
    nisa.dma_copy(src=expert_index_sb, dst=_hbm_tiled_store_view(expert_index, 0, num_t_tiles_total, P_MAX))
    return [router_logits, expert_index, expert_affinities]


def router_topk_sharded_sbuf_index_torch_wrapper(
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
):
    """Torch golden for the sharded SBUF-index wrapper (standard [T,E]/[T,k] outputs)."""
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
        router_pre_norm=False,
        shard_on_tokens=True,
    )


def router_topk_layout1_odd_kernel_wrapper(
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
):
    """Feed a layout-1 SBUF x built with the unbalanced (odd num_h_tiles) split.

    router_topk_input_x_load's sb_layout=1 path requires an even num_h_tiles
    (H % 256 == 0), so it cannot produce this input. We build the SBUF layout
    directly, mirroring router_topk_input_w_load's odd-num_h_tiles halves so the
    x and w H-index mappings agree and the matmul is correct.
    """
    T, H = x.shape
    num_h_tiles = H // P_MAX
    h2_first = num_h_tiles // 2
    h2_second = num_h_tiles - h2_first
    x_sb = nl.ndarray((P_MAX, T, num_h_tiles), dtype=x.dtype, buffer=nl.sbuf)
    # Half 0: H rows [0, h2_first*128) -> [128, T, h2_first], H-index = p*h2_first + tile
    x_half0 = (
        x.slice(dim=1, start=0, end=h2_first * P_MAX).reshape_dim(dim=1, shape=[P_MAX, h2_first]).permute([1, 0, 2])
    )
    nisa.dma_copy(src=x_half0, dst=x_sb.slice(dim=2, start=0, end=h2_first))
    # Half 1: H rows [h2_first*128, H) -> [128, T, h2_second], H-index = h2_first*128 + p*h2_second + tile
    x_half1 = (
        x.slice(dim=1, start=h2_first * P_MAX, end=H).reshape_dim(dim=1, shape=[P_MAX, h2_second]).permute([1, 0, 2])
    )
    nisa.dma_copy(src=x_half1, dst=x_sb.slice(dim=2, start=h2_first, end=num_h_tiles))
    router_topk(
        x=x_sb,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=XSBLayout_tp2013__1,
        router_pre_norm=False,
    )
    return [router_logits, expert_index, expert_affinities]


def router_topk_layout1_odd_torch_wrapper(
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
):
    """Torch golden for the layout-1 odd-num_h_tiles wrapper (x is [T, H])."""
    return router_topk_torch_ref(
        x=x,
        w=w,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=k,
        x_hbm_layout=1,
        x_sb_layout=x_sb_layout,
        router_pre_norm=False,
    )
