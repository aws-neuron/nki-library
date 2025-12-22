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

"""Quantized projection operations for output projection CTE kernel with FP8 static quantization."""

from typing import List, Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.tensor_view import TensorView
from .output_projection_cte_parameters import P_MAX, QuantizationConfig, TilingConfig
from .output_projection_cte_tensor_io import (
    get_zero_bias_vector_sbuf,
    invert_static_quant_scales,
    load_bias,
    load_input_tensor_quantized,
    load_quantized_weights,
    load_static_quant_input_scales,
    load_static_quant_weight_scales,
)

# FP8 typical max positive range
_FP8_MAX_POS_RANGE = 240.0


def perform_static_quantized_projection(
    attention_hbm: nl.ndarray,
    weight_hbm: nl.ndarray,
    output_hbm: nl.ndarray,
    bias_hbm: Optional[nl.ndarray],
    input_scale_hbm: nl.ndarray,
    weight_scale_hbm: nl.ndarray,
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Perform FP8 static quantized output projection with double row matmul.

    Follows same loop structure as float projection with quantization and
    double row matmul optimizations.

    Tiling Strategy:
        Input: [B, N, D, S] tiled on S with tile_size=512, subtile_size=128
        Weight: [N, D, H] tiled on H with tile_size based on SBUF budget
        Output: [B, S, H] written per s_subtile

        Memory Budget:
        - Weight SBUF: n_size * d_size * h_block_size * 1 byte (FP8) <= 10MB
        - Attention SBUF: n_size * d_size * s_tile_size * dtype_size (bf16/fp16)
        - Quantized Attention SBUF: n_size * d_size * s_tile_size * 1 byte (FP8)
        - Result SBUF: P_MAX * h_block_size * dtype_size per subtile

    Args:
        attention_hbm (nl.ndarray): [B, N, D, S], Input attention tensor.
        weight_hbm (nl.ndarray): [N, D, H], Quantized weight tensor.
        output_hbm (nl.ndarray): [B, S, H], Output tensor.
        bias_hbm (Optional[nl.ndarray]): [1, H], Optional bias tensor.
        input_scale_hbm (nl.ndarray): [128, 1], Input quantization scales.
        weight_scale_hbm (nl.ndarray): [128, 1], Weight quantization scales.
        prg_id (int): Program ID for LNC sharding.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes results to output_hbm tensor.

    Notes:
        - Uses double row matmul for FP8 quantization when N is even.
        - Loop order: h_block -> batch -> s_block.
    """
    input_scale_sbuf = load_static_quant_input_scales(input_scale_hbm)
    weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
    invert_static_quant_scales(input_scale_sbuf)

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        weight_view = TensorView(weight_hbm).slice(dim=2, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf_list = load_quantized_weights(weight_view=weight_view, cfg=cfg, quant_config=quant_config)

        bias_sbuf = None
        if bias_hbm is not None:
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
                    TensorView(output_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
                )

                _process_quantized_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    bias_sbuf=bias_sbuf,
                    input_scale_sbuf=input_scale_sbuf,
                    weight_scale_sbuf=weight_scale_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    cfg=cfg,
                    quant_config=quant_config,
                )


def _process_quantized_batch_tile(
    attention_view: TensorView,
    output_view: TensorView,
    w_sbuf_list: List[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    input_scale_sbuf: nl.ndarray,
    weight_scale_sbuf: nl.ndarray,
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Process a single batch tile with quantization and double row matmul.

    Args:
        attention_view (TensorView): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        output_view (TensorView): View of output tensor for current batch/h_block [S, h_block_size].
        w_sbuf_list (List[nl.ndarray]): List of quantized weight tensors in SBUF.
        bias_sbuf (Optional[nl.ndarray]): Bias tensor in SBUF.
        input_scale_sbuf (nl.ndarray): Input quantization scales in SBUF.
        weight_scale_sbuf (nl.ndarray): Weight quantization scales in SBUF.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes results to output tensor via output_view.
    """
    curr_s_tile_size = attention_view.shape[2]
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1: Load attention tensors
    attention_sb = load_input_tensor_quantized(
        attention_view=attention_view,
        cfg=cfg,
        quant_config=quant_config,
    )

    # Step 2: Quantize attention tensors
    quant_attention_sb = _quantize_attention_tensors(
        attention_sb=attention_sb,
        input_scale_sbuf=input_scale_sbuf,
        curr_s_tile_size=curr_s_tile_size,
        cfg=cfg,
        quant_config=quant_config,
    )

    # Step 3: Compute matmul and dequantize
    result_sb = _compute_matmul_dequantize(
        quant_attention_sb=quant_attention_sb,
        w_sbuf_list=w_sbuf_list,
        bias_sbuf=bias_sbuf,
        weight_scale_sbuf=weight_scale_sbuf,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        attention_dtype=attention_view.dtype,
        cfg=cfg,
        quant_config=quant_config,
    )

    # Step 4: Write results to output
    _write_results_to_output(
        result_sb=result_sb,
        output_view=output_view,
        s_start=s_start,
        s_block_idx=s_block_idx,
        curr_h_block_size=curr_h_block_size,
        cfg=cfg,
    )


def _quantize_attention_tensors(
    attention_sb: List[nl.ndarray],
    input_scale_sbuf: nl.ndarray,
    curr_s_tile_size: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """
    Quantize loaded attention tensors to FP8.

    Args:
        attention_sb (List[nl.ndarray]): Loaded attention tensors in SBUF.
        input_scale_sbuf (nl.ndarray): Inverted input scales in SBUF.
        curr_s_tile_size (int): Current S tile size.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.ndarray]: Quantized attention tensors in SBUF.
    """
    num_heads_to_process = cfg.n_size // 2 if quant_config.use_double_row else cfg.n_size
    quant_attention_sb = []

    for head_idx in range(num_heads_to_process):
        if quant_config.use_double_row:
            quant_tensor = nl.ndarray(
                (cfg.d_size, 2, cfg.s_tile.tile_size),
                dtype=quant_config.quant_data_type,
                buffer=nl.sbuf,
            )
            _perform_input_static_quantization(
                input_sbuf=attention_sb[head_idx][: cfg.d_size, 0:1, :curr_s_tile_size],
                inverse_input_scale_sbuf=input_scale_sbuf,
                quant_res_sbuf=quant_tensor[: cfg.d_size, 0:1, :curr_s_tile_size],
            )
            _perform_input_static_quantization(
                input_sbuf=attention_sb[head_idx][: cfg.d_size, 1:2, :curr_s_tile_size],
                inverse_input_scale_sbuf=input_scale_sbuf,
                quant_res_sbuf=quant_tensor[: cfg.d_size, 1:2, :curr_s_tile_size],
            )
        else:
            quant_tensor = nl.ndarray(
                (cfg.d_size, cfg.s_tile.tile_size),
                dtype=quant_config.quant_data_type,
                buffer=nl.sbuf,
            )
            _perform_input_static_quantization(
                input_sbuf=attention_sb[head_idx][: cfg.d_size, :curr_s_tile_size],
                inverse_input_scale_sbuf=input_scale_sbuf,
                quant_res_sbuf=quant_tensor[: cfg.d_size, :curr_s_tile_size],
            )
        quant_attention_sb.append(quant_tensor)

    return quant_attention_sb


def _compute_matmul_dequantize(
    quant_attention_sb: List[nl.ndarray],
    w_sbuf_list: List[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    weight_scale_sbuf: nl.ndarray,
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    attention_dtype,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """
    Compute matmul across heads and dequantize results.

    Args:
        quant_attention_sb (List[nl.ndarray]): Quantized attention tensors in SBUF.
        w_sbuf_list (List[nl.ndarray]): Quantized weight tensors in SBUF.
        bias_sbuf (Optional[nl.ndarray]): Bias tensor in SBUF.
        weight_scale_sbuf (nl.ndarray): Weight scales for dequantization.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        curr_h_block_size (int): Current H block size.
        attention_dtype: Data type for output.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.ndarray]: Result tensors in SBUF after matmul and dequantization.
    """
    num_heads_to_process = cfg.n_size // 2 if quant_config.use_double_row else cfg.n_size

    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        result_sb.append(nl.ndarray((P_MAX, curr_h_block_size), dtype=attention_dtype, buffer=nl.sbuf))

    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        s_subtile_start = cfg.s_tile.get_local_subtile_start(s_subtile_idx)

        for h_subtile_idx in range(cfg.h_tile.subtile_dim_info.tile_count):
            curr_h_subtile_size = cfg.h_tile.get_local_subtile_bound(h_block_idx, h_subtile_idx)
            if curr_h_subtile_size <= 0:
                break
            h_subtile_start = cfg.h_tile.get_local_subtile_start(h_subtile_idx)

            res_psum = nl.ndarray(
                (curr_s_subtile_size, curr_h_subtile_size),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            # Accumulate matmul across heads
            for head_idx in range(num_heads_to_process):
                if quant_config.use_double_row:
                    attention_slice = quant_attention_sb[head_idx][
                        : cfg.d_size,
                        0:2,
                        s_subtile_start : s_subtile_start + curr_s_subtile_size,
                    ]
                    weight_slice = w_sbuf_list[head_idx][
                        : cfg.d_size,
                        0:2,
                        h_subtile_start : h_subtile_start + curr_h_subtile_size,
                    ]
                    nisa.nc_matmul(
                        res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                        attention_slice,
                        weight_slice,
                        perf_mode=nisa.matmul_perf_mode.double_row,
                    )
                else:
                    attention_slice = quant_attention_sb[head_idx][
                        : cfg.d_size,
                        s_subtile_start : s_subtile_start + curr_s_subtile_size,
                    ]
                    weight_slice = w_sbuf_list[head_idx][
                        : cfg.d_size,
                        h_subtile_start : h_subtile_start + curr_h_subtile_size,
                    ]
                    nisa.nc_matmul(
                        res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                        attention_slice,
                        weight_slice,
                    )

            # Dequantize using weight scales
            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            nisa.activation(
                dst=result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                op=nl.copy,
                data=res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                scale=weight_scale_sbuf[:curr_s_subtile_size, :1],
                bias=get_zero_bias_vector_sbuf(P_MAX)[:curr_s_subtile_size, :1],
            )

            if bias_sbuf is not None:
                nisa.tensor_tensor(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    nl.add,
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


def _perform_input_static_quantization(
    input_sbuf: nl.ndarray,
    inverse_input_scale_sbuf: nl.ndarray,
    quant_res_sbuf: nl.ndarray,
) -> None:
    """
    Quantize input activation using scales.

    Args:
        input_sbuf (nl.ndarray): Input tensor in SBUF (2D or 3D for double-row).
        inverse_input_scale_sbuf (nl.ndarray): Inverted input scales in SBUF.
        quant_res_sbuf (nl.ndarray): Output quantized tensor in SBUF (same shape as input).

    Returns:
        None: Writes quantized result to quant_res_sbuf.
    """
    d_size = input_sbuf.shape[0]
    bias_vector = get_zero_bias_vector_sbuf(d_size)

    # Scale input by inverse scale, then clamp to FP8 range
    nisa.activation(
        dst=input_sbuf,
        op=nl.copy,
        data=input_sbuf,
        scale=inverse_input_scale_sbuf[:d_size, :1],
        bias=bias_vector,
    )
    nisa.tensor_scalar(
        dst=quant_res_sbuf,
        data=input_sbuf,
        op0=nl.minimum,
        operand0=_FP8_MAX_POS_RANGE,
        op1=nl.maximum,
        operand1=-_FP8_MAX_POS_RANGE,
    )
