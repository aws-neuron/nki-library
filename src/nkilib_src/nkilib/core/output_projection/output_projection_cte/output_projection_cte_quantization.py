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

"""Quantized projection operations for output projection CTE kernel with FP8 and MX quantization."""

from typing import List, Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_helpers import get_max_positive_value_for_dtype
from .output_projection_cte_parameters import (
    P_MAX,
    QuantizationConfig,
    TilingConfig,
    _q_height,
    _q_width,
)
from .output_projection_cte_tensor_io import (
    create_constant_mx_scales,
    get_zero_bias_vector_sbuf,
    invert_static_quant_scales,
    load_bias,
    load_input_tensor_quantized,
    load_mx_compact_weight_scales,
    load_mx_input_interleaved,
    load_mx_prequantized_input,
    load_mx_quantized_weights,
    load_mx_weight_scales,
    load_quantized_weights,
    load_row_weight_dequant_scales,
    load_static_quant_input_scales,
    load_static_quant_weight_scales,
)


def perform_static_quantized_projection(
    attention_hbm: nl.NkiTensor,
    weight_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    bias_hbm: Optional[nl.NkiTensor],
    input_scale_hbm: nl.NkiTensor,
    weight_scale_hbm: nl.NkiTensor,
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
        attention_hbm (nl.NkiTensor): [B, N, D, S], Input attention tensor.
        weight_hbm (nl.NkiTensor): [N, D, H], Quantized weight tensor.
        output_hbm (nl.NkiTensor): [B, S, H], Output tensor.
        bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        input_scale_hbm (nl.NkiTensor): [128, 1], Input quantization scales.
        weight_scale_hbm (nl.NkiTensor): [128, 1], Weight quantization scales.
        prg_id (int): Program ID for LNC sharding.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes results to output_hbm tensor.

    Notes:
        - Uses double row matmul for FP8 quantization when N is even.
        - Loop order: h_block -> batch -> s_block.
    """
    weight_hbm = weight_hbm.reshape((cfg.n_size, cfg.d_size, cfg.h_size))

    if cfg.group_size > 1 or quant_config.use_double_row:
        attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.n_size, cfg.d_size, cfg.s_size))

    input_scale_sbuf = load_static_quant_input_scales(input_scale_hbm)
    weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
    invert_static_quant_scales(input_scale_sbuf)

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        weight_view = weight_hbm.slice(dim=2, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf_list = load_quantized_weights(weight_view=weight_view, cfg=cfg, quant_config=quant_config)

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = bias_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                    dim=2, start=s_start, end=s_start + curr_s_tile_size
                )
                output_view = output_hbm.select(dim=0, index=batch_idx).slice(
                    dim=1, start=h_start, end=h_start + curr_h_block_size
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
    attention_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    input_scale_sbuf: nl.NkiTensor,
    weight_scale_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Process a single batch tile with quantization and double row matmul.

    Args:
        attention_view (NkiTensor): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        output_view (NkiTensor): View of output tensor for current batch/h_block [S, h_block_size].
        w_sbuf_list (List[nl.NkiTensor]): List of quantized weight tensors in SBUF.
        bias_sbuf (Optional[nl.NkiTensor]): Bias tensor in SBUF.
        input_scale_sbuf (nl.NkiTensor): Input quantization scales in SBUF.
        weight_scale_sbuf (nl.NkiTensor): Weight quantization scales in SBUF.
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
    attention_sb: List[nl.NkiTensor],
    input_scale_sbuf: nl.NkiTensor,
    curr_s_tile_size: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """
    Quantize loaded attention tensors to FP8.

    Args:
        attention_sb (List[nl.NkiTensor]): Loaded attention tensors in SBUF.
        input_scale_sbuf (nl.NkiTensor): Inverted input scales in SBUF.
        curr_s_tile_size (int): Current S tile size.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Quantized attention tensors in SBUF.
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
                quant_config=quant_config,
            )
            _perform_input_static_quantization(
                input_sbuf=attention_sb[head_idx][: cfg.d_size, 1:2, :curr_s_tile_size],
                inverse_input_scale_sbuf=input_scale_sbuf,
                quant_res_sbuf=quant_tensor[: cfg.d_size, 1:2, :curr_s_tile_size],
                quant_config=quant_config,
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
                quant_config=quant_config,
            )
        quant_attention_sb.append(quant_tensor)

    return quant_attention_sb


def _compute_matmul_dequantize(
    quant_attention_sb: List[nl.NkiTensor],
    w_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    weight_scale_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    attention_dtype,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """
    Compute matmul across heads and dequantize results.

    Args:
        quant_attention_sb (List[nl.NkiTensor]): Quantized attention tensors in SBUF.
        w_sbuf_list (List[nl.NkiTensor]): Quantized weight tensors in SBUF.
        bias_sbuf (Optional[nl.NkiTensor]): Bias tensor in SBUF.
        weight_scale_sbuf (nl.NkiTensor): Weight scales for dequantization.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        curr_h_block_size (int): Current H block size.
        attention_dtype: Data type for output.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Result tensors in SBUF after matmul and dequantization.
    """
    num_heads_to_process = cfg.n_size // 2 if quant_config.use_double_row else cfg.n_size

    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
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

            # Dequantize using weight scales - alternate between scalar and vector engines
            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            engine = nisa.scalar_engine if (s_subtile_idx + h_subtile_idx) % 2 == 0 else nisa.vector_engine
            nisa.tensor_scalar(
                dst=result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                data=res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                op0=nl.multiply,
                operand0=weight_scale_sbuf[:curr_s_subtile_size, :1],
                engine=engine,
            )

            if bias_sbuf != None:
                nisa.tensor_tensor(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    nl.add,
                )

    return result_sb


def _write_results_to_output(
    result_sb: List[nl.NkiTensor],
    output_view: nl.NkiTensor,
    s_start: int,
    s_block_idx: int,
    curr_h_block_size: int,
    cfg: TilingConfig,
) -> None:
    """
    Write result tensors to output HBM.

    Args:
        result_sb (List[nl.NkiTensor]): Result tensors in SBUF.
        output_view (NkiTensor): View of output tensor.
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
        nisa.dma_copy(out_subtile_view, result_sb[s_subtile_idx][:curr_s_subtile_size, :curr_h_block_size])


def _perform_input_static_quantization(
    input_sbuf: nl.NkiTensor,
    inverse_input_scale_sbuf: nl.NkiTensor,
    quant_res_sbuf: nl.NkiTensor,
    quant_config: QuantizationConfig,
) -> None:
    """
    Quantize input activation using scales.

    Args:
        input_sbuf (nl.NkiTensor): Input tensor in SBUF (2D or 3D for double-row).
        inverse_input_scale_sbuf (nl.NkiTensor): Inverted input scales in SBUF.
        quant_res_sbuf (nl.NkiTensor): Output quantized tensor in SBUF (same shape as input).
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes quantized result to quant_res_sbuf.
    """
    d_size = input_sbuf.shape[0]
    bias_vector = get_zero_bias_vector_sbuf(d_size)
    max_val = get_max_positive_value_for_dtype(quant_config.weight_data_type)

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
        operand0=max_val,
        op1=nl.maximum,
        operand1=-max_val,
    )


# ============================================================================
# Row FP8 Quantization Functions (TRN2)
# ============================================================================

_FP8_E4M3_MAX = 240.0
_ROW_QUANT_MIN_SCALE = 1e-5


def perform_row_quantized_projection(
    attention_hbm: nl.NkiTensor,
    weight_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    bias_hbm: Optional[nl.NkiTensor],
    weight_scale_hbm: nl.NkiTensor,
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Perform ROW FP8 quantized output projection for TRN2.

    Input is [B, S, N*D] bf16, dynamically row-quantized per token on-device.
    Weights are FP8 with per-row dequant scales [P_MAX, H].

    Flow per tile:
    1. Load attention [S_tile, D] per head (S=partition, D=free)
    2. Per-token quantize: absmax over D, scale, clamp to FP8
    3. nc_transpose to [D, S] for matmul
    4. nc_matmul(attn[D, S], weight[D, H]) -> [S, H]
    5. Dequant: tensor_tensor(psum * weight_row_scale) then activation(result * input_dequant_scale)

    Args:
        attention_hbm: [B, S, N*D], Input attention tensor (bf16).
        weight_hbm: [N*D, H], FP8 weight tensor.
        output_hbm: [B, S, H], Output tensor.
        bias_hbm: Optional [1, H] bias tensor.
        weight_scale_hbm: [P_MAX, H], Per-row weight dequant scale.
        prg_id: Program ID for LNC sharding.
        cfg: Tiling configuration.
        quant_config: Quantization configuration.
    """
    weight_hbm = weight_hbm.reshape((cfg.n_size, cfg.d_size, cfg.h_size))
    # Track original N, D from attention shape (cfg values may differ due to double row)
    n_orig = attention_hbm.shape[2]
    d_orig = attention_hbm.shape[3]

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        weight_view = weight_hbm.slice(dim=2, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf_list = load_quantized_weights(weight_view=weight_view, cfg=cfg, quant_config=quant_config)

        weight_row_scale_sbuf = load_row_weight_dequant_scales(
            weight_scale_hbm,
            h_start,
            curr_h_block_size,
            cfg.h_tile.tile_size,
        )

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = bias_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                # attention_hbm is [B, S, N, D] — select batch, slice S
                attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                    dim=0, start=s_start, end=s_start + curr_s_tile_size
                )
                output_view = output_hbm.select(dim=0, index=batch_idx).slice(
                    dim=1, start=h_start, end=h_start + curr_h_block_size
                )

                _process_row_quantized_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    bias_sbuf=bias_sbuf,
                    weight_row_scale_sbuf=weight_row_scale_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    n_orig=n_orig,
                    d_orig=d_orig,
                    cfg=cfg,
                    quant_config=quant_config,
                )


def _process_row_quantized_batch_tile(
    attention_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    weight_row_scale_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    n_orig: int,
    d_orig: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """Process a single batch tile with ROW FP8 quantization.

    Loads attention as [S_sub, N*D] bf16, row-quantizes, dma_transposes per head
    to [D_orig, S_sub], reshapes to [D, 2, S] for double row. Falls back to [D, S]
    when double row is disabled.

    Args:
        attention_view: View of attention [curr_s_tile, N_orig, D_orig].
        output_view: View of output [S, curr_h_block_size].
        w_sbuf_list: Quantized weight tensors in SBUF.
        bias_sbuf: Optional bias tensor in SBUF.
        weight_row_scale_sbuf: [P_MAX, h_tile_size], Per-row weight dequant scale.
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        n_orig: Original number of heads.
        d_orig: Original head dimension.
        cfg: Tiling configuration.
        quant_config: Quantization configuration.
    """
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1: Load [S_sub, N*D] bf16, row-quantize, dma_transpose per orig head, reshape for matmul

    h_dim = n_orig * d_orig  # N*D (original, same as cfg.n_size * cfg.d_size)
    num_matmul_heads = cfg.n_size // 2 if quant_config.use_double_row else cfg.n_size
    quant_attention_sb = []
    global_dequant_scales = []

    # Reshape attention_view from [curr_s_tile, N_orig, D_orig] to [curr_s_tile, N*D]
    attn_flat_view = attention_view.reshape((attention_view.shape[0], h_dim))

    # Pre-allocate transposed buffers per original head
    # Double row: FP8 (nc_transpose path), non-double-row: bf16 (dma_transpose path)
    xpose_dtype = quant_config.quant_data_type if quant_config.use_double_row else nl.bfloat16
    transposed_heads = []
    for head_idx in range(n_orig):
        transposed_heads.append(nl.ndarray((d_orig, cfg.s_tile.tile_size), dtype=xpose_dtype, buffer=nl.sbuf))

    for s_sub_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_sub = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_sub_idx)
        if curr_s_sub <= 0:
            break
        s_sub_start = cfg.s_tile.get_local_subtile_start(s_sub_idx)

        # Load [S_sub, N*D] bf16
        attn_sub = nl.ndarray((P_MAX, h_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        sub_view = attn_flat_view.slice(dim=0, start=s_sub_start, end=s_sub_start + curr_s_sub)
        nisa.dma_copy(attn_sub[:curr_s_sub, :h_dim], sub_view)

        # Row-quantize: absmax, scale, clamp — stays bf16
        quant_sub, dequant_scale = _perform_input_row_quantization(
            input_sbuf=attn_sub[:curr_s_sub, :h_dim],
            quant_dtype=quant_config.quant_data_type,
        )
        global_dequant_scales.append(dequant_scale)

        quant_nd = quant_sub.reshape((curr_s_sub, n_orig, d_orig))
        for head_idx in range(n_orig):
            if quant_config.use_double_row:
                # Double row: cast to FP8, nc_transpose with step-of-2 PSUM (HW constraint for 1-byte dtype)
                fp8_head = nl.ndarray((curr_s_sub, d_orig), dtype=quant_config.quant_data_type, buffer=nl.sbuf)
                nisa.tensor_copy(dst=fp8_head, src=quant_nd[:curr_s_sub, head_idx, :d_orig])

                _FP8_PSUM_STEP = 2
                xpose_psum = nl.ndarray(
                    (d_orig, curr_s_sub, _FP8_PSUM_STEP),
                    dtype=quant_config.quant_data_type,
                    buffer=nl.psum,
                )
                nisa.nc_transpose(
                    dst=xpose_psum.select(dim=2, index=0),
                    data=fp8_head,
                )
                nisa.tensor_copy(
                    dst=transposed_heads[head_idx][:d_orig, s_sub_start : s_sub_start + curr_s_sub],
                    src=xpose_psum[:d_orig, :curr_s_sub, 0],
                )
            else:
                # No double row: dma_transpose bf16 [S_sub, D_orig] -> [D_orig, S_sub]
                nisa.dma_transpose(
                    dst=transposed_heads[head_idx][:d_orig, s_sub_start : s_sub_start + curr_s_sub],
                    src=quant_nd[:curr_s_sub, head_idx, :d_orig],
                )

    # Pass transposed heads — for double row, pack into [D, 2, S]
    if quant_config.use_double_row:
        if n_orig == cfg.n_size // 2:
            # D was split: n_orig heads, each [D_orig, S] -> split into [D, 2, S]
            for head_idx in range(n_orig):
                packed_sb = nl.ndarray((cfg.d_size, 2, cfg.s_tile.tile_size), dtype=xpose_dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 0:1, : cfg.s_tile.tile_size],
                    src=transposed_heads[head_idx][: cfg.d_size, : cfg.s_tile.tile_size],
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 1:2, : cfg.s_tile.tile_size],
                    src=transposed_heads[head_idx][cfg.d_size : d_orig, : cfg.s_tile.tile_size],
                )
                quant_attention_sb.append(packed_sb)
        else:
            # N is even, pair heads: [D, S] + [D, S] -> [D, 2, S]
            for pair_idx in range(n_orig // 2):
                packed_sb = nl.ndarray((cfg.d_size, 2, cfg.s_tile.tile_size), dtype=xpose_dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 0:1, : cfg.s_tile.tile_size],
                    src=transposed_heads[pair_idx * 2][: cfg.d_size, : cfg.s_tile.tile_size],
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 1:2, : cfg.s_tile.tile_size],
                    src=transposed_heads[pair_idx * 2 + 1][: cfg.d_size, : cfg.s_tile.tile_size],
                )
                quant_attention_sb.append(packed_sb)
    else:
        quant_attention_sb = transposed_heads

    input_dequant_scale_sb = [global_dequant_scales]

    # Step 2: Compute matmul and dequantize
    result_sb = _compute_row_matmul_dequantize(
        quant_attention_sb=quant_attention_sb,
        w_sbuf_list=w_sbuf_list,
        bias_sbuf=bias_sbuf,
        weight_row_scale_sbuf=weight_row_scale_sbuf,
        input_dequant_scale_sb=input_dequant_scale_sb,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        attention_dtype=nl.bfloat16,
        cfg=cfg,
        quant_config=quant_config,
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


def _perform_input_row_quantization(
    input_sbuf: nl.NkiTensor,
    quant_dtype,
) -> tuple:
    """Row-quantize input per token: absmax over free dim, scale, quantize.

    Args:
        input_sbuf: [S, H] input tensor in SBUF (S=partition, H=free).
        quant_dtype: Target FP8 dtype for max value lookup.

    Returns:
        Tuple of (quantized [S, H], dequant_scale [S, 1]).
    """
    s_size = input_sbuf.shape[0]
    h_size = input_sbuf.shape[1]
    max_val = get_max_positive_value_for_dtype(quant_dtype)

    # absmax = max(abs(input), dim=-1) → fp32[S, 1]
    absmax = nl.ndarray((s_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    abs_sbuf = nl.ndarray((s_size, h_size), dtype=input_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(
        dst=abs_sbuf,
        data=input_sbuf,
        op0=nl.abs,
        operand0=0.0,
        reduce_op=nl.maximum,
        reduce_res=absmax,
    )

    # dequant_scale = max(absmax / fp8_max, MINVAL) → fp32[S, 1]
    dequant_scale = nl.ndarray((s_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dequant_scale,
        data=absmax,
        op0=nl.multiply,
        operand0=1.0 / max_val,
        op1=nl.maximum,
        operand1=_ROW_QUANT_MIN_SCALE,
    )

    # quant_scale = 1 / dequant_scale → fp32[S, 1]
    quant_scale = nl.ndarray((s_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=quant_scale, data=dequant_scale)

    # quantized = input * quant_scale → [S, H], quant_scale [S, 1] broadcasts over H
    quantized = nl.ndarray((s_size, h_size), dtype=input_sbuf.dtype, buffer=nl.sbuf)
    nisa.activation(
        dst=quantized,
        op=nl.copy,
        data=input_sbuf,
        scale=quant_scale,
    )

    return quantized, dequant_scale


def _compute_row_matmul_dequantize(
    quant_attention_sb: List[nl.NkiTensor],
    w_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    weight_row_scale_sbuf: nl.NkiTensor,
    input_dequant_scale_sb: List[nl.NkiTensor],
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    attention_dtype,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """Compute matmul across heads and apply two-step ROW dequant.

    After matmul accumulation:
    1. tensor_tensor: multiply by per-row weight dequant scale [S, H]
    2. activation: multiply by per-token input dequant scale [S, 1]
    3. Optional bias add

    Args:
        quant_attention_sb: Quantized attention per head [D, S] in SBUF.
        w_sbuf_list: Quantized weight tensors per head in SBUF.
        bias_sbuf: Optional bias tensor.
        weight_row_scale_sbuf: [P_MAX, h_tile_size], Per-row weight dequant scale.
        input_dequant_scale_sb: Per-head input dequant scales [S, 1].
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        curr_h_block_size: Current H block size.
        attention_dtype: Output data type.
        cfg: Tiling configuration.
        quant_config: Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Result tensors per s_subtile.
    """
    zero_bias = get_zero_bias_vector_sbuf(P_MAX)

    # input_dequant_scale_sb is [head][s_subtile] -> [P_MAX, 1]
    # Use first head's scales (all heads have same per-token scale)
    input_dequant_scales = input_dequant_scale_sb[0]

    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
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
            num_matmul_heads = cfg.n_size // 2 if quant_config.use_double_row else cfg.n_size
            for head_idx in range(num_matmul_heads):
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

            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            dst_tile = result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)]

            # Dequant step 1: multiply by per-row weight dequant scale [S, H]
            nisa.tensor_tensor(
                dst=dst_tile,
                data1=res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                data2=weight_row_scale_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                op=nl.multiply,
            )

            # Dequant step 2: multiply by per-token input dequant scale [S, 1]
            nisa.activation(
                dst=dst_tile,
                op=nl.copy,
                data=dst_tile,
                scale=input_dequant_scales[s_subtile_idx][:curr_s_subtile_size, :1],
                bias=zero_bias[:curr_s_subtile_size, :1],
            )

            # Add bias if present
            if bias_sbuf != None:
                nisa.tensor_tensor(
                    dst=dst_tile,
                    data1=dst_tile,
                    data2=bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    op=nl.add,
                )

    return result_sb


# ============================================================================
# MX Quantization Functions
# ============================================================================


def perform_mx_quantized_projection(
    attention_hbm: nl.NkiTensor,
    weight_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    bias_hbm: Optional[nl.NkiTensor],
    weight_scale_hbm: nl.NkiTensor,
    input_scale_hbm: Optional[nl.NkiTensor],
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Perform MX quantized output projection with on-device or pre-quantized input.

    Supports two modes based on quant_config.input_quantized:
    - Online quantization (input_quantized=False): Input is bf16/fp16, quantized on-device
    - Pre-quantized (input_quantized=True): Input is already float8_e4m3fn_x4 with scales

    Tiling Strategy:
        Input (online): [B, D/512, 128_D, 4_D, S] tiled on D with tile_size=128
        Input (pre-quantized): [B, D_packed, S], 4_D x4 packed in dtype, tiled on D with tile_size=128
        Weight: [D, H] tiled on H with tile_size based on SBUF budget
        Output: [B, S, H] written per s_subtile (128 elements)

        Loop order: h_block -> batch -> s_block
        Inner processing: Load input -> (Quantize if online) -> Matmul -> Write

    Padding Strategy:
        When D % 128 == 96 (e.g., D=96), the last d_tile is padded to 128:
        - nc_matmul_mx only supports partition dims of 32, 64, 128
        - PaddedTileInfo.get_bounds() returns (padded=128, actual=96)
        - Zero-padding ensures padded elements don't affect matmul results

    Memory Budget:
        - Weight SBUF: d_tile_count * d_tile_size * h_block_size * 1 byte (float4_e2m1fn_x4, pre-packed)
        - Weight Scale SBUF: d_tile_count * (d_tile_size // _q_height) * h_block_size bytes (uint8, 1 scale per 32 FP4 elements)
        - Attention SBUF: d_tile_count * d_tile_size * s_tile_size * _q_width * 2 bytes (bf16, before quantization)
        - Quantized Attention: d_tile_count * d_tile_size * s_tile_size * 2 bytes (float8_e4m3fn_x4, _q_width packed)
        - Attention Scale: d_tile_count * d_tile_size * s_tile_size * 1 byte (uint8, same shape as quantized attention)
        - Result SBUF: s_subtile_count * P_MAX * h_block_size * 2 bytes (bf16)

    Args:
        attention_hbm (nl.NkiTensor): Input attention tensor in HBM.
            - Online: [B, D/512, 128_D, 4_D, S] (bf16/fp16)
            - Pre-quantized: [B, D_packed, S], 4_D x4 packed in dtype (float8_e4m3fn_x4)
        weight_hbm (nl.NkiTensor): [D, H], Pre-quantized weights (float4_e2m1fn_x4).
        output_hbm (nl.NkiTensor): [B, S, H], Output tensor.
        bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        weight_scale_hbm (nl.NkiTensor): [D//_q_height, H], Weight scales for MX.
        input_scale_hbm (Optional[nl.NkiTensor]): [B, D//_q_height, S], Input scales for pre-quantized mode.
        prg_id (int): Program ID for LNC sharding.
        cfg (TilingConfig): Tiling config with PaddedTileInfo for d_tile.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes results to output_hbm tensor.
    """
    # Reshape attention based on whether input is pre-quantized
    if quant_config.input_quantized:
        # Pre-quantized: [B, D, S*_q_width(but packed as S)]
        attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.d_tile.get_actual_dim_size(), cfg.s_size))
        input_scale_hbm = input_scale_hbm.reshape(
            (cfg.b_size, cfg.d_tile.get_actual_dim_size() // _q_height, cfg.s_size)
        )
    else:
        # Online quantization: [B, D, _q_width, S] where _q_width is the interleaved dimension
        attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.d_tile.get_actual_dim_size(), _q_width, cfg.s_size))
    for h_block_idx in range(cfg.h_tile.tile_count):
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        weight_view = weight_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)

        w_sbuf_list = load_mx_quantized_weights(weight_view, quant_config.quant_data_type, cfg, quant_config)
        if quant_config.compact_weight_scales:
            # block-128: load compact [D//128, H//128] and expand to
            # the dense MX hardware layout per d_tile.
            w_scale_sbuf_list = load_mx_compact_weight_scales(
                weight_scale_hbm=weight_scale_hbm,
                h_start=h_start,
                curr_h_block_size=curr_h_block_size,
                cfg=cfg,
            )
        else:
            weight_scale_view = weight_scale_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            w_scale_sbuf_list = load_mx_weight_scales(weight_scale_view, cfg)

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = bias_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                # Slice attention based on input format
                if quant_config.input_quantized:
                    # Pre-quantized: [D, S*_q_width(but packed as S)] - slice on dim=1 (S dimension)
                    attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                        dim=1, start=s_start, end=s_start + curr_s_tile_size
                    )
                    input_scale_view = input_scale_hbm.select(dim=0, index=batch_idx).slice(
                        dim=1, start=s_start, end=s_start + curr_s_tile_size
                    )
                else:
                    # Online quantization: [D, _q_width, S] - slice on dim=2 (S dimension)
                    attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                        dim=2, start=s_start, end=s_start + curr_s_tile_size
                    )
                    input_scale_view = None

                output_view = output_hbm.select(dim=0, index=batch_idx).slice(
                    dim=1, start=h_start, end=h_start + curr_h_block_size
                )

                _process_mx_quantized_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    w_scale_sbuf_list=w_scale_sbuf_list,
                    bias_sbuf=bias_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    cfg=cfg,
                    quant_config=quant_config,
                    input_scale_view=input_scale_view,
                )


def _process_mx_quantized_batch_tile(
    attention_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
    input_scale_view: Optional[nl.NkiTensor] = None,
) -> None:
    """Process a single batch tile with MX quantization and matmul.

    Supports two modes based on quant_config.input_quantized:
    - Online quantization: Load bf16 input, transpose, quantize on-device
    - Pre-quantized: Load pre-quantized float8_e4m3fn_x4 input and scales directly

    Args:
        attention_view: View of attention tensor.
            - Online: [D, _q_width, curr_s_tile_size]
            - Pre-quantized: [D, curr_s_tile_size*_q_width(but packed as curr_s_tile_size)]
        output_view: View of output tensor [S, curr_h_block_size].
        w_sbuf_list: Pre-loaded weight tensors per d_tile.
        w_scale_sbuf_list: Pre-loaded weight scales per d_tile.
        bias_sbuf: Bias tensor in SBUF.
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        cfg: Tiling configuration.
        quant_config: Quantization configuration.
        input_scale_view: View of input scales [D//_q_height, curr_s_tile_size] for pre-quantized mode.
    """
    # Get S tile size from appropriate dimension based on input format
    if quant_config.input_quantized:
        curr_s_tile_size = attention_view.shape[1]  # [D, S*4(but packed as S)]
    else:
        curr_s_tile_size = attention_view.shape[2]  # [D, _q_width, S]

    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1 & 2: Load and quantize attention (branched based on input format)
    if quant_config.input_quantized:
        # Pre-quantized path: load quantized input and scales directly
        quant_attn_list, attn_scale_list, _ = load_mx_prequantized_input(attention_view, input_scale_view, cfg)
    else:
        # Online quantization path: load, transpose, then quantize
        attn_sbuf_list, padded_s_tile_size = load_mx_input_interleaved(attention_view, cfg, quant_config)
        quant_attn_list, attn_scale_list = _quantize_mx_attention_tensors(
            attn_sbuf_list=attn_sbuf_list,
            padded_s_tile_size=padded_s_tile_size,
            cfg=cfg,
        )

    # Step 3: Compute MX matmul accumulating across d_tiles
    result_sb = _compute_mx_matmul(
        quant_attn_list=quant_attn_list,
        attn_scale_list=attn_scale_list,
        w_sbuf_list=w_sbuf_list,
        w_scale_sbuf_list=w_scale_sbuf_list,
        bias_sbuf=bias_sbuf,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        attention_dtype=output_view.dtype,
        cfg=cfg,
    )

    # Step 4: Write results to output HBM
    _write_results_to_output(
        result_sb=result_sb,
        output_view=output_view,
        s_start=s_start,
        s_block_idx=s_block_idx,
        curr_h_block_size=curr_h_block_size,
        cfg=cfg,
    )


def _quantize_mx_attention_tensors(
    attn_sbuf_list: List[nl.NkiTensor],
    padded_s_tile_size: int,
    cfg: TilingConfig,
) -> Tuple[List[nl.NkiTensor], List[nl.NkiTensor]]:
    """Quantize attention tensors using nisa.quantize_mx.

    Args:
        attn_sbuf_list: Loaded attention [d_tile_count] -> [d_tile_size, padded_S, _q_width].
        padded_s_tile_size: Padded S tile size (even).
        cfg: Tiling configuration.

    Returns:
        Tuple of (quant_attn_list, attn_scale_list).
    """
    quant_attn_list = []
    attn_scale_list = []
    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        curr_d_tile_size, _ = cfg.d_tile.get_bounds(d_tile_idx)
        if curr_d_tile_size <= 0:
            break

        attn_2d = attn_sbuf_list[d_tile_idx].reshape((curr_d_tile_size, padded_s_tile_size * _q_width))
        quant_attn_sbuf = nl.ndarray((curr_d_tile_size, padded_s_tile_size), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        attn_scale_sbuf = nl.ndarray((curr_d_tile_size, padded_s_tile_size), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(
            src=attn_2d[:, : padded_s_tile_size * _q_width],
            dst=quant_attn_sbuf[:, :padded_s_tile_size],
            dst_scale=attn_scale_sbuf[:, :padded_s_tile_size],
        )
        quant_attn_list.append(quant_attn_sbuf)
        attn_scale_list.append(attn_scale_sbuf)

    return quant_attn_list, attn_scale_list


def _compute_mx_matmul(
    quant_attn_list: List[nl.NkiTensor],
    attn_scale_list: List[nl.NkiTensor],
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf_list: List[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    attention_dtype,
    cfg: TilingConfig,
) -> List[nl.NkiTensor]:
    """Compute MX matmul across d_tiles and apply optional bias.

    Args:
        quant_attn_list: Quantized attention per d_tile.
        attn_scale_list: Attention scales per d_tile.
        w_sbuf_list: Weight tensors per d_tile.
        w_scale_sbuf_list: Weight scales per d_tile.
        bias_sbuf: Optional bias tensor.
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        curr_h_block_size: Current H block size.
        attention_dtype: Data type for output.
        cfg: Tiling configuration.

    Returns:
        Result tensors [s_subtile_count] -> [P_MAX, curr_h_block_size].
    """
    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        result_sb.append(nl.ndarray((P_MAX, curr_h_block_size), dtype=attention_dtype, buffer=nl.sbuf))

    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        s_subtile_start = cfg.s_tile.get_local_subtile_start(s_subtile_idx)
        # nc_matmul_mx requires even number of elements in free dimension
        padded_s_subtile_size = curr_s_subtile_size + (curr_s_subtile_size % 2)

        for h_subtile_idx in range(cfg.h_tile.subtile_dim_info.tile_count):
            curr_h_subtile_size = cfg.h_tile.get_local_subtile_bound(h_block_idx, h_subtile_idx)
            if curr_h_subtile_size <= 0:
                break
            h_subtile_start = cfg.h_tile.get_local_subtile_start(h_subtile_idx)

            res_psum = nl.ndarray((padded_s_subtile_size, curr_h_subtile_size), dtype=nl.bfloat16, buffer=nl.psum)

            # Accumulate matmul across all d_tiles
            for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
                padded_d_size, actual_d_size = cfg.d_tile.get_bounds(d_tile_idx)
                attn_slice = quant_attn_list[d_tile_idx][:, s_subtile_start : s_subtile_start + padded_s_subtile_size]
                attn_scale_slice = attn_scale_list[d_tile_idx][
                    :, s_subtile_start : s_subtile_start + padded_s_subtile_size
                ]
                w_slice = w_sbuf_list[d_tile_idx][:, h_subtile_start : h_subtile_start + curr_h_subtile_size]
                w_scale_slice = w_scale_sbuf_list[d_tile_idx][
                    :, h_subtile_start : h_subtile_start + curr_h_subtile_size
                ]

                nisa.nc_matmul_mx(
                    dst=res_psum[:padded_s_subtile_size, :curr_h_subtile_size],
                    stationary=attn_slice,
                    moving=w_slice,
                    stationary_scale=attn_scale_slice,
                    moving_scale=w_scale_slice,
                )

            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            if bias_sbuf != None:
                nisa.tensor_tensor(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                    bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    nl.add,
                )
            else:
                # Alternate between scalar and vector engines to maximize throughput
                nisa.tensor_copy(
                    src=res_psum[:curr_s_subtile_size, 0:curr_h_subtile_size],
                    dst=result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    engine=nisa.scalar_engine if h_subtile_idx % 2 == 0 else nisa.vector_engine,
                )

    return result_sb


# ============================================================================
# Static MX FP8 Quantization Functions
# ============================================================================


def perform_static_mx_quantized_projection(
    attention_hbm: nl.NkiTensor,
    weight_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    bias_hbm: Optional[nl.NkiTensor],
    input_scale_hbm: nl.NkiTensor,
    weight_scale_hbm: nl.NkiTensor,
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Perform static MX FP8 quantized output projection using nc_matmul_mx.

    Uses static [128, 1] scales like STATIC quantization but performs matmul
    using nc_matmul_mx with packed x4 format. The MX scales are constant (126)
    and actual dequantization uses the static scales.

    Tiling Strategy:
        Input: [B, N, D, S] tiled on S with tile_size=512, subtile_size=128
        Weight: [N*D, H] tiled on H with tile_size based on SBUF budget
        Output: [B, S, H] written per s_subtile

        Loop order: h_block -> batch -> s_block

    Args:
        attention_hbm (nl.NkiTensor): [B, N, D, S], Input attention tensor.
        weight_hbm (nl.NkiTensor): [N*D, H], Pre-quantized weights (float8_e4m3fn_x4).
        output_hbm (nl.NkiTensor): [B, S, H], Output tensor.
        bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        input_scale_hbm (nl.NkiTensor): [128, 1], Input quantization scales.
        weight_scale_hbm (nl.NkiTensor): [128, 1], Weight quantization scales.
        prg_id (int): Program ID for LNC sharding.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        None: Writes results to output_hbm tensor.
    """
    # Load static scales (reuse from static path)
    input_scale_sbuf = load_static_quant_input_scales(input_scale_hbm)
    weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
    invert_static_quant_scales(input_scale_sbuf)

    # Create zero bias vector once for reuse
    zero_bias_sbuf = get_zero_bias_vector_sbuf(P_MAX)

    # Create constant MX scales (127)
    w_scale_sbuf = create_constant_mx_scales(cfg.d_tile.tile_info.tile_size, cfg.h_tile.subtile_dim_info.tile_size)
    attn_scale_sbuf = create_constant_mx_scales(cfg.d_tile.tile_info.tile_size, cfg.s_tile.subtile_dim_info.tile_size)

    # Reshape attention to [B, D, 4, S] for interleaved loading (4 heads packed)
    attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.n_size * cfg.d_size // _q_width, _q_width, cfg.s_size))
    # assuming weights are reshaped and permuted offline
    weight_hbm = weight_hbm.reshape((cfg.n_size * cfg.d_size // _q_width, cfg.h_size, _q_width))
    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        # Load pre-quantized weights as FP8
        weight_view = weight_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf_list = load_mx_quantized_weights(weight_view, nl.float8_e4m3fn, cfg, quant_config)

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = bias_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                # [D, _q_width, curr_s_tile_size] for interleaved loading
                attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                    dim=2, start=s_start, end=s_start + curr_s_tile_size
                )
                output_view = output_hbm.select(dim=0, index=batch_idx).slice(
                    dim=1, start=h_start, end=h_start + curr_h_block_size
                )

                _process_static_mx_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    w_scale_sbuf=w_scale_sbuf,
                    attn_scale_sbuf=attn_scale_sbuf,
                    bias_sbuf=bias_sbuf,
                    input_scale_sbuf=input_scale_sbuf,
                    weight_scale_sbuf=weight_scale_sbuf,
                    zero_bias_sbuf=zero_bias_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    cfg=cfg,
                    quant_config=quant_config,
                )


def _process_static_mx_batch_tile(
    attention_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf: nl.NkiTensor,
    attn_scale_sbuf: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    input_scale_sbuf: nl.NkiTensor,
    weight_scale_sbuf: nl.NkiTensor,
    zero_bias_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Process a single batch tile with static MX quantization.

    Args:
        attention_view (NkiTensor): View of attention [D//4, 4, curr_s_tile_size].
        output_view (NkiTensor): View of output [S, curr_h_block_size].
        w_sbuf_list (List[nl.NkiTensor]): Weight tensors per d_tile.
        w_scale_sbuf (nl.NkiTensor): Constant MX weight scales.
        attn_scale_sbuf (nl.NkiTensor): Constant MX attention scales.
        bias_sbuf (Optional[nl.NkiTensor]): Bias tensor in SBUF.
        input_scale_sbuf (nl.NkiTensor): Inverted input scales for quantization.
        weight_scale_sbuf (nl.NkiTensor): Combined dequantization scales.
        zero_bias_sbuf (nl.NkiTensor): Pre-allocated zero bias vector.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.
    """
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1: Load attention tensors using interleaved loader [D, _q_width, S] -> [D, S, _q_width]
    attn_sbuf_list, padded_s_tile_size = load_mx_input_interleaved(attention_view, cfg, quant_config)

    # Step 2: Quantize to FP8 and pack to x4 format
    quant_attn_list = _quantize_static_mx_attention(
        attn_sbuf_list=attn_sbuf_list,
        input_scale_sbuf=input_scale_sbuf,
        zero_bias_sbuf=zero_bias_sbuf,
        padded_s_tile_size=padded_s_tile_size,
        cfg=cfg,
        quant_config=quant_config,
    )

    # Step 3: Compute MX matmul and dequantize with static scales
    result_sb = _compute_static_mx_matmul(
        quant_attn_list=quant_attn_list,
        attn_scale_sbuf=attn_scale_sbuf,
        w_sbuf_list=w_sbuf_list,
        w_scale_sbuf=w_scale_sbuf,
        bias_sbuf=bias_sbuf,
        weight_scale_sbuf=weight_scale_sbuf,
        zero_bias_sbuf=zero_bias_sbuf,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        padded_s_tile_size=padded_s_tile_size,
        cfg=cfg,
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


def _quantize_static_mx_attention(
    attn_sbuf_list: List[nl.NkiTensor],
    input_scale_sbuf: nl.NkiTensor,
    zero_bias_sbuf: nl.NkiTensor,
    padded_s_tile_size: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """
    Quantize attention to FP8 and pack to x4 format for static MX.

    Uses same engine strategy as load_mx_input_interleaved for optimal pipelining:
    - d_tile_count >= 2: Alternate engines across d_tiles. This allows load+quantize
      for d_tile N to pipeline with load+quantize for d_tile N+1 on different engines.
    - d_tile_count == 1: Use vector engine (scalar engine handled the load's second half).

    Args:
        attn_sbuf_list (List[nl.NkiTensor]): Attention tensors per d_tile [D, S, 4].
        input_scale_sbuf (nl.NkiTensor): Inverted input scales.
        zero_bias_sbuf (nl.NkiTensor): Pre-allocated zero bias vector.
        padded_s_tile_size (int): Padded S tile size.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Quantized attention tensors per d_tile.
    """
    quant_attn_list = []
    quant_dtype = quant_config.weight_data_type
    max_val = get_max_positive_value_for_dtype(quant_dtype)
    use_pipeline_strategy = cfg.d_tile.tile_info.tile_count >= 2

    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        curr_d_tile_size, _ = cfg.d_tile.get_bounds(d_tile_idx)
        if curr_d_tile_size <= 0:
            break
        attn_sbuf = attn_sbuf_list[d_tile_idx].reshape(
            (curr_d_tile_size, padded_s_tile_size * _q_width)
        )  # [D, S, _q_width]

        # Quantize to FP8: scale by inverse input scale
        # Use same engine as load to enable pipelining across d_tiles
        if use_pipeline_strategy:
            engine = nisa.vector_engine if d_tile_idx % 2 == 0 else nisa.scalar_engine
        else:
            engine = nisa.vector_engine
        nisa.tensor_scalar(
            dst=attn_sbuf,
            data=attn_sbuf,
            op0=nl.multiply,
            operand0=input_scale_sbuf[:curr_d_tile_size, :1],
            engine=engine,
        )

        # Clamp to FP8 range and convert to float8_e4m3
        quant_attn_sbuf = nl.ndarray(
            (curr_d_tile_size, padded_s_tile_size * _q_width), dtype=quant_dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=quant_attn_sbuf,
            data=attn_sbuf,
            op0=nl.minimum,
            operand0=max_val,
            op1=nl.maximum,
            operand1=-max_val,
        )
        quant_attn_list.append(quant_attn_sbuf)
    return quant_attn_list


def _compute_static_mx_matmul(
    quant_attn_list: List[nl.NkiTensor],
    attn_scale_sbuf: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    weight_scale_sbuf: nl.NkiTensor,
    zero_bias_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    padded_s_tile_size: int,
    cfg: TilingConfig,
) -> List[nl.NkiTensor]:
    """
    Compute MX matmul and dequantize with static scales.

    Uses alternating scalar/vector engines for dequantization to balance load.

    Args:
        quant_attn_list (List[nl.NkiTensor]): Quantized attention per d_tile (x4 packed).
        attn_scale_sbuf (nl.NkiTensor): Constant MX attention scales.
        w_sbuf_list (List[nl.NkiTensor]): Weight tensors per d_tile.
        w_scale_sbuf (nl.NkiTensor): Constant MX weight scales.
        bias_sbuf (Optional[nl.NkiTensor]): Bias tensor.
        weight_scale_sbuf (nl.NkiTensor): Combined dequantization scale.
        zero_bias_sbuf (nl.NkiTensor): Pre-allocated zero bias vector.
        s_block_idx (int): Current S block index.
        h_block_idx (int): Current H block index.
        curr_h_block_size (int): Current H block size.
        padded_s_tile_size (int): Padded S tile size (for access pattern).
        cfg (TilingConfig): Tiling configuration.

    Returns:
        List[nl.NkiTensor]: Result tensors per s_subtile.
    """
    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        result_sb.append(nl.ndarray((P_MAX, curr_h_block_size), dtype=nl.bfloat16, buffer=nl.sbuf))

    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        s_subtile_start = cfg.s_tile.get_local_subtile_start(s_subtile_idx)
        padded_s_subtile_size = curr_s_subtile_size + (curr_s_subtile_size % 2)

        for h_subtile_idx in range(cfg.h_tile.subtile_dim_info.tile_count):
            curr_h_subtile_size = cfg.h_tile.get_local_subtile_bound(h_block_idx, h_subtile_idx)
            if curr_h_subtile_size <= 0:
                break
            h_subtile_start = cfg.h_tile.get_local_subtile_start(h_subtile_idx)

            res_psum = nl.ndarray((padded_s_subtile_size, curr_h_subtile_size), dtype=nl.bfloat16, buffer=nl.psum)
            # Accumulate matmul across all d_tiles
            for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
                curr_d_tile_size, _ = cfg.d_tile.get_bounds(d_tile_idx)
                # quant_attn_list has shape [D, S*4] as float8_e4m3fn, use .view() to reinterpret as [D, S] float8_e4m3fn_x4
                attn_slice = quant_attn_list[d_tile_idx].view(nl.float8_e4m3fn_x4)[
                    :, s_subtile_start : s_subtile_start + padded_s_subtile_size
                ]
                attn_scale_slice = attn_scale_sbuf[:, :padded_s_subtile_size]
                # w_sbuf_list has shape [D, H*4] as float8_e4m3fn, use .view() to reinterpret as [D, H] float8_e4m3fn_x4
                w_slice = w_sbuf_list[d_tile_idx].view(nl.float8_e4m3fn_x4)[
                    :, h_subtile_start : h_subtile_start + curr_h_subtile_size
                ]
                w_scale_slice = w_scale_sbuf[:, :curr_h_subtile_size]
                nisa.nc_matmul_mx(
                    dst=res_psum[:padded_s_subtile_size, :curr_h_subtile_size],
                    stationary=attn_slice,
                    moving=w_slice,
                    stationary_scale=attn_scale_slice,
                    moving_scale=w_scale_slice,
                )

            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            # Dequantize with static scales - alternate between scalar and vector engines
            engine = nisa.scalar_engine if (s_subtile_idx + h_subtile_idx) % 2 == 0 else nisa.vector_engine
            nisa.tensor_scalar(
                dst=result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                data=res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                op0=nl.multiply,
                operand0=weight_scale_sbuf[:curr_s_subtile_size, :1],
                engine=engine,
            )

            if bias_sbuf != None:
                nisa.tensor_tensor(
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    nl.add,
                )

    return result_sb


# ============================================================================
# Row MX FP8 Quantization Functions
# ============================================================================


def perform_row_mx_quantized_projection(
    attention_hbm: nl.NkiTensor,
    weight_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    bias_hbm: Optional[nl.NkiTensor],
    input_scale_hbm: nl.NkiTensor,
    weight_scale_hbm: nl.NkiTensor,
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """
    Perform ROW_MX quantized output projection.

    Input is MXFP4 quantized on-device (same as MX path), weights are row FP8
    quantized with per-row dequant scales. Uses nc_matmul_mx with real MX scales
    for attention and constant 127 MX scales for weights (dummy), then applies
    per-row weight dequant scale after matmul.

    Args:
        attention_hbm (nl.NkiTensor): [B, N, D, S], Input attention tensor (bf16).
        weight_hbm (nl.NkiTensor): [N*D, H], Pre-quantized FP8 weights (float8_e4m3fn).
        output_hbm (nl.NkiTensor): [B, S, H], Output tensor.
        bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        input_scale_hbm (nl.NkiTensor): Unused for ROW_MX (input quantized on-device).
        weight_scale_hbm (nl.NkiTensor): [128, H], Per-row weight dequant scale.
        prg_id (int): Program ID for LNC sharding.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.
    """
    # Create constant MX scales (127) for weights (dummy — actual dequant is per-row)
    w_scale_sbuf = create_constant_mx_scales(cfg.d_tile.tile_info.tile_size, cfg.h_tile.subtile_dim_info.tile_size)

    # Reshape attention to [B, D, _q_width, S] for interleaved loading
    attention_hbm = attention_hbm.reshape((cfg.b_size, cfg.d_tile.get_actual_dim_size(), _q_width, cfg.s_size))
    # Reshape weights: pre-shuffled [N*D//4, H, 4]
    weight_hbm = weight_hbm.reshape((cfg.n_size * cfg.d_size // _q_width, cfg.h_size, _q_width))

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        # Load FP8 weights (same loader as STATIC_MX)
        weight_view = weight_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
        w_sbuf_list = load_mx_quantized_weights(weight_view, nl.float8_e4m3fn, cfg, quant_config)

        # Load per-row weight dequant scales for this h_block
        weight_row_scale_sbuf = load_row_weight_dequant_scales(
            weight_scale_hbm,
            h_start,
            curr_h_block_size,
            cfg.h_tile.tile_size,
        )

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = bias_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                attention_view = attention_hbm.select(dim=0, index=batch_idx).slice(
                    dim=2, start=s_start, end=s_start + curr_s_tile_size
                )
                output_view = output_hbm.select(dim=0, index=batch_idx).slice(
                    dim=1, start=h_start, end=h_start + curr_h_block_size
                )

                _process_row_mx_batch_tile(
                    attention_view=attention_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    w_scale_sbuf=w_scale_sbuf,
                    bias_sbuf=bias_sbuf,
                    weight_row_scale_sbuf=weight_row_scale_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    cfg=cfg,
                    quant_config=quant_config,
                )


def _process_row_mx_batch_tile(
    attention_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    weight_row_scale_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    """Process a single batch tile with ROW_MX quantization.

    Input is MXFP4 quantized on-device, weights are FP8 with constant 127 MX scales.
    After matmul, applies per-row weight dequant.

    Args:
        attention_view: View of attention [D//4, 4, curr_s_tile_size].
        output_view: View of output [S, curr_h_block_size].
        w_sbuf_list: Weight tensors per d_tile.
        w_scale_sbuf: Constant MX weight scales (127).
        bias_sbuf: Optional bias tensor in SBUF.
        weight_row_scale_sbuf: [P_MAX, h_tile_size], Per-row weight dequant scale.
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        cfg: Tiling configuration.
        quant_config: Quantization configuration.
    """
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    # Step 1: Load and transpose attention for MX quantization
    attn_sbuf_list, padded_s_tile_size = load_mx_input_interleaved(attention_view, cfg, quant_config)

    # Step 2: Quantize attention using quantize_mx (real MX scales)
    quant_attn_list, attn_scale_list = _quantize_mx_attention_tensors(
        attn_sbuf_list=attn_sbuf_list,
        padded_s_tile_size=padded_s_tile_size,
        cfg=cfg,
    )

    # Step 3: Compute MX matmul with constant 127 weight scales, then ROW_MX dequant
    result_sb = _compute_row_mx_matmul(
        quant_attn_list=quant_attn_list,
        attn_scale_list=attn_scale_list,
        w_sbuf_list=w_sbuf_list,
        w_scale_sbuf=w_scale_sbuf,
        bias_sbuf=bias_sbuf,
        weight_row_scale_sbuf=weight_row_scale_sbuf,
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        padded_s_tile_size=padded_s_tile_size,
        cfg=cfg,
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


def _compute_row_mx_matmul(
    quant_attn_list: List[nl.NkiTensor],
    attn_scale_list: List[nl.NkiTensor],
    w_sbuf_list: List[nl.NkiTensor],
    w_scale_sbuf: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    weight_row_scale_sbuf: nl.NkiTensor,
    s_block_idx: int,
    h_block_idx: int,
    curr_h_block_size: int,
    padded_s_tile_size: int,
    cfg: TilingConfig,
) -> List[nl.NkiTensor]:
    """Compute MX matmul with FP8 x4 weights and per-row weight dequant.

    Matmul uses nc_matmul_mx with real attention MX scales and constant 127
    weight MX scales. After matmul:
    1. tensor_tensor: multiply by per-row weight dequant scale (per H column)
    2. Optionally add bias

    Args:
        quant_attn_list: Quantized attention per d_tile (from quantize_mx).
        attn_scale_list: Attention MX scales per d_tile (from quantize_mx).
        w_sbuf_list: FP8 weight tensors per d_tile.
        w_scale_sbuf: Constant MX weight scales (127).
        bias_sbuf: Optional bias tensor.
        weight_row_scale_sbuf: [P_MAX, h_tile_size], Per-row weight dequant scale.
        s_block_idx: Current S block index.
        h_block_idx: Current H block index.
        curr_h_block_size: Current H block size.
        padded_s_tile_size: Padded S tile size (for the .view() reinterpret).
        cfg: Tiling configuration.

    Returns:
        List[nl.NkiTensor]: Result tensors per s_subtile.
    """
    result_sb = []
    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        result_sb.append(nl.ndarray((P_MAX, curr_h_block_size), dtype=nl.bfloat16, buffer=nl.sbuf))

    for s_subtile_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_subtile_size = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_subtile_idx)
        if curr_s_subtile_size <= 0:
            break
        s_subtile_start = cfg.s_tile.get_local_subtile_start(s_subtile_idx)
        padded_s_subtile_size = curr_s_subtile_size + (curr_s_subtile_size % 2)

        for h_subtile_idx in range(cfg.h_tile.subtile_dim_info.tile_count):
            curr_h_subtile_size = cfg.h_tile.get_local_subtile_bound(h_block_idx, h_subtile_idx)
            if curr_h_subtile_size <= 0:
                break
            h_subtile_start = cfg.h_tile.get_local_subtile_start(h_subtile_idx)

            res_psum = nl.ndarray((padded_s_subtile_size, curr_h_subtile_size), dtype=nl.bfloat16, buffer=nl.psum)

            # Accumulate matmul across all d_tiles (same as STATIC_MX)
            for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
                curr_d_tile_size, _ = cfg.d_tile.get_bounds(d_tile_idx)
                # Attention: real MX quantized, [D, S] as float8_e4m3fn_x4
                attn_slice = quant_attn_list[d_tile_idx][:, s_subtile_start : s_subtile_start + padded_s_subtile_size]
                attn_scale_slice = attn_scale_list[d_tile_idx][
                    :, s_subtile_start : s_subtile_start + padded_s_subtile_size
                ]
                # Weights: FP8 x4 packed, use .view() to reinterpret
                w_slice = w_sbuf_list[d_tile_idx].view(nl.float8_e4m3fn_x4)[
                    :, h_subtile_start : h_subtile_start + curr_h_subtile_size
                ]
                w_scale_slice = w_scale_sbuf[:, :curr_h_subtile_size]

                nisa.nc_matmul_mx(
                    dst=res_psum[:padded_s_subtile_size, :curr_h_subtile_size],
                    stationary=attn_slice,
                    moving=w_slice,
                    stationary_scale=attn_scale_slice,
                    moving_scale=w_scale_slice,
                )

            h_indices = h_subtile_idx * cfg.h_tile.subtile_dim_info.tile_size
            dst_tile = result_sb[s_subtile_idx][:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)]

            # ROW_MX dequant: multiply by per-row weight dequant scale
            nisa.tensor_tensor(
                dst=dst_tile,
                data1=res_psum[:curr_s_subtile_size, :curr_h_subtile_size],
                data2=weight_row_scale_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                op=nl.multiply,
            )

            # Add bias if present
            if bias_sbuf != None:
                nisa.tensor_tensor(
                    dst=dst_tile,
                    data1=dst_tile,
                    data2=bias_sbuf[:curr_s_subtile_size, nl.ds(h_indices, curr_h_subtile_size)],
                    op=nl.add,
                )

    return result_sb
