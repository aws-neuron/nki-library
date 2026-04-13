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

"""Float (non-quantized) bf16/fp16/fp32 projection for output projection CTE kernel."""

from typing import List, Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.tensor_view import TensorView
from .output_projection_cte_parameters import P_MAX, TilingConfig
from .output_projection_cte_tensor_io import (
    load_bias,
    load_float_weights,
    load_input_tensor_float,
)


def perform_float_projection(
    attention_hbm: nl.ndarray,
    weight_hbm: nl.ndarray,
    bias_hbm: Optional[nl.ndarray],
    out_hbm: nl.ndarray,
    cfg: TilingConfig,
    prg_id: int,
) -> None:
    """
    Perform float (non-quantized) output projection: out = attention @ weight + bias.

    Handles standard bf16/fp16/fp32 matmul path without quantization.

    Tiling Strategy:
        Input: [B, N, D, S] tiled on S with tile_size=512, subtile_size=128
        Weight: [N, D, H] tiled on H with tile_size based on SBUF budget
        Output: [B, S, H] written per s_subtile

        Memory Budget:
        - Weight SBUF: n_size * d_size * h_block_size * dtype_size <= 10MB
        - Attention SBUF: n_size * d_size * s_tile_size * dtype_size
        - Result SBUF: P_MAX * h_block_size * dtype_size per subtile

    Args:
        attention_hbm (nl.ndarray): [B, N, D, S], Input attention tensor in HBM.
        weight_hbm (nl.ndarray): [N, D, H], Weight tensor in HBM (reshaped in main kernel).
        bias_hbm (Optional[nl.ndarray]): [1, H], Optional bias tensor in HBM.
        out_hbm (nl.ndarray): [B, S, H], Output tensor in HBM to write results.
        cfg (TilingConfig): Tiling configuration with dimension sizes.
        prg_id (int): Program ID for LNC sharding.

    Returns:
        None: Writes results to out tensor.

    Notes:
        - Iterates h_blocks in outer loop to limit SBUF usage for weights.
        - Reloads attention scores for each h_block.
    """
    weight_hbm = weight_hbm.reshape((cfg.n_size, cfg.d_size, cfg.h_size))

    if cfg.group_size > 1:
        attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.n_size, cfg.d_size, cfg.s_size))

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        weight_view = TensorView(weight_hbm).slice(dim=2, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf = load_float_weights(weight_view=weight_view, cfg=cfg, weight_dtype=weight_hbm.dtype)

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = TensorView(bias_hbm).slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                attention_view = (
                    TensorView(attention_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=2, start=s_start, end=s_start + curr_s_tile_size)
                )
                output_view = (
                    TensorView(out_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
                )

                _process_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf=w_sbuf,
                    bias_sbuf=bias_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    cfg=cfg,
                    weight_dtype=weight_hbm.dtype,
                )


def _process_batch_tile(
    attention_view: TensorView,
    output_view: TensorView,
    w_sbuf: List[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    weight_dtype,
) -> None:
    """
    Process a single batch tile for one h_block: computes attention @ weight + bias.

    Args:
        attention_view (TensorView): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        output_view (TensorView): View of output tensor for current batch/h_block [S, h_block_size].
        w_sbuf (List[nl.ndarray]): List of weight tensors in SBUF (one per head).
        bias_sbuf (Optional[nl.ndarray]): Bias tensor in SBUF.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        cfg (TilingConfig): Tiling configuration.
        weight_dtype: Weight tensor dtype (attention is cast to this to avoid mixed precision errors).

    Returns:
        None: Writes results to output tensor via output_view.
    """
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1: Load attention tensors (cast to weight dtype to avoid mixed precision matmul error)
    attention_sb = load_input_tensor_float(attention_view=attention_view, cfg=cfg, target_dtype=weight_dtype)

    # Step 2: Compute matmul and add bias
    result_sb = _compute_matmul_add_bias(
        attention_sb=attention_sb,
        w_sbuf=w_sbuf,
        bias_sbuf=bias_sbuf,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        attention_dtype=attention_view.dtype,
        cfg=cfg,
    )

    # Step 3: Write results to output
    _write_results_to_output(
        result_sb=result_sb,
        output_view=output_view,
        s_start=s_start,
        s_block_idx=s_block_idx,
        curr_h_block_size=curr_h_block_size,
        cfg=cfg,
    )


def _compute_matmul_add_bias(
    attention_sb: List[nl.ndarray],
    w_sbuf: List[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    attention_dtype,
    cfg: TilingConfig,
) -> List[nl.ndarray]:
    """
    Compute matmul across heads and add bias.

    Args:
        attention_sb (List[nl.ndarray]): Attention tensors in SBUF.
        w_sbuf (List[nl.ndarray]): Weight tensors in SBUF.
        bias_sbuf (Optional[nl.ndarray]): Bias tensor in SBUF.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        curr_h_block_size (int): Current H block size.
        attention_dtype: Data type for output.
        cfg (TilingConfig): Tiling configuration.

    Returns:
        List[nl.ndarray]: Result tensors in SBUF after matmul and bias addition.
    """
    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        result_sb.append(nl.ndarray((P_MAX, curr_h_block_size), dtype=attention_dtype, buffer=nl.sbuf))

    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        s_subtile_start = cfg.s_tile.get_local_subtile_start(s_subtile_idx)
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        num_h_subtiles = cfg.h_tile.get_actual_subtile_num(h_block_idx)
        for h_subtile_idx in range(num_h_subtiles):
            h_subtile_start = cfg.h_tile.get_local_subtile_start(h_subtile_idx)
            curr_h_subtile_size = cfg.h_tile.get_local_subtile_bound(h_block_idx, h_subtile_idx)
            if curr_h_subtile_size <= 0:
                break

            res_psum = nl.ndarray(
                (curr_s_subtile_size, curr_h_subtile_size),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            # Accumulate matmul across all heads
            for head_idx in range(cfg.n_size):
                attention_slice = attention_sb[head_idx][
                    : cfg.d_size,
                    s_subtile_start : s_subtile_start + curr_s_subtile_size,
                ]
                weight_slice = w_sbuf[head_idx][
                    : cfg.d_size,
                    h_subtile_start : h_subtile_start + curr_h_subtile_size,
                ]
                nisa.nc_matmul(
                    res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                    attention_slice,
                    weight_slice,
                )

            # Add bias or copy result
            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            if bias_sbuf != None:
                nisa.tensor_tensor(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                    bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    nl.add,
                )
            else:
                nisa.tensor_copy(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                    engine=nisa.scalar_engine if s_subtile_idx % 2 == 0 else nisa.vector_engine,
                )

    return result_sb


def _write_results_to_output(
    result_sb: List[nl.ndarray],
    output_view: TensorView,
    s_start: int,
    s_block_idx: int,
    curr_h_block_size: int,
    cfg: TilingConfig,
) -> None:
    """
    Write result tensors to output HBM.

    Args:
        result_sb (List[nl.ndarray]): Result tensors in SBUF.
        output_view (TensorView): View of output tensor.
        s_start (int): Start offset in S dimension.
        s_block_idx (int): Current S block index.
        curr_h_block_size (int): Current H block size.
        cfg (TilingConfig): Tiling configuration.
    """
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        s_offset = s_start + s_subtile_idx * P_MAX

        out_subtile_view = output_view.slice(dim=0, start=s_offset, end=s_offset + curr_s_subtile_size)
        nisa.dma_copy(out_subtile_view.get_view(), result_sb[s_subtile_idx][:curr_s_subtile_size, :curr_h_block_size])
