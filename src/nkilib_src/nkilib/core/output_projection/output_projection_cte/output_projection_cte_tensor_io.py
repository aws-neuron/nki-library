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

"""Tensor I/O operations for output projection CTE kernel including weights, biases, and scales."""

from typing import List, Tuple

import nki.isa as nisa
import nki.language as nl

from ...utils.tensor_view import TensorView
from .output_projection_cte_parameters import _SBUF_QUADRANT_SIZE, QuantizationConfig, TilingConfig, _q_height, _q_width


def get_zero_bias_vector_sbuf(tile_size: int, activation_data_type=nl.float32) -> nl.ndarray:
    """
    Create zero bias vector in SBUF.

    Args:
        tile_size (int): Size of the bias vector.
        activation_data_type: Data type for the bias vector.

    Returns:
        nl.ndarray: [tile_size, 1], Zero-initialized bias vector in SBUF.
    """
    bias_vector_sbuf = nl.ndarray((tile_size, 1), dtype=activation_data_type, buffer=nl.sbuf)
    nisa.memset(bias_vector_sbuf, value=0.0, engine=nisa.gpsimd_engine)
    return bias_vector_sbuf


def invert_static_quant_scales(scales_sbuf: nl.ndarray) -> None:
    """
    Invert scales for quantization (reciprocal).

    Args:
        scales_sbuf (nl.ndarray): Scales tensor in SBUF to invert in-place.

    Returns:
        None: Modifies scales_sbuf in-place.
    """
    nisa.reciprocal(scales_sbuf, scales_sbuf)


def load_static_quant_input_scales(static_quant_scale_hbm: nl.ndarray) -> nl.ndarray:
    """
    Load and prepare input quantization scales.

    Args:
        static_quant_scale_hbm (nl.ndarray): [P_MAX, 1], Input scales tensor in HBM.

    Returns:
        nl.ndarray: [P_MAX, 1], Input scales in SBUF.
    """
    P_MAX = nl.tile_size.pmax
    static_quant_scale_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    # Layout: [P_MAX, 1] where P_MAX=128 partition elements, 1 free dimension
    scale_view = TensorView(static_quant_scale_hbm).slice(dim=0, start=0, end=P_MAX)
    nisa.dma_copy(dst=static_quant_scale_sbuf, src=scale_view.get_view())
    return static_quant_scale_sbuf


def load_static_quant_weight_scales(
    static_quant_weight_scale_hbm: nl.ndarray,
    static_quant_input_scale_sbuf: nl.ndarray,
) -> nl.ndarray:
    """
    Load weight scales and multiply with input scales for dequantization.

    Args:
        static_quant_weight_scale_hbm (nl.ndarray): [P_MAX, 1], Weight scales tensor in HBM.
        static_quant_input_scale_sbuf (nl.ndarray): [P_MAX, 1], Input scales in SBUF.

    Returns:
        nl.ndarray: [P_MAX, 1], Combined weight scales in SBUF (weight_scale * input_scale).
    """
    P_MAX = nl.tile_size.pmax
    bias_vector = get_zero_bias_vector_sbuf(P_MAX)
    static_quant_weight_scale_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Layout: [P_MAX, 1] where P_MAX=128 partition elements, 1 free dimension
    weight_scale_view = TensorView(static_quant_weight_scale_hbm).slice(dim=0, start=0, end=P_MAX)
    nisa.dma_copy(dst=static_quant_weight_scale_sbuf, src=weight_scale_view.get_view())

    # Combined scale = weight_scale * input_scale (for dequantization)
    nisa.activation(
        dst=static_quant_weight_scale_sbuf,
        op=nl.copy,
        data=static_quant_weight_scale_sbuf,
        scale=static_quant_input_scale_sbuf,
        bias=bias_vector,
    )

    return static_quant_weight_scale_sbuf


def load_bias(
    bias_view: TensorView,
    cfg: TilingConfig,
) -> nl.ndarray:
    """
    Load bias into SBUF and broadcast to [P_MAX, h_block_size].

    Uses tensorview.broadcast on the HBM tensor and loads directly into the
    broadcasted array for better DMA utilization.

    Args:
        bias_view (TensorView): View of bias tensor [1, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.

    Returns:
        nl.ndarray: [P_MAX, h_block_size], Broadcasted bias tensor in SBUF.

    Notes:
        - If curr_h_block_size < h_block_size, tensor contains garbage at end.
        - DMA operations only use valid elements.
    """
    P_MAX = nl.tile_size.pmax
    curr_h_block_size = bias_view.shape[1]

    # Broadcast bias_view from [1, curr_h_block_size] to [P_MAX, curr_h_block_size]
    broadcasted_bias_view = bias_view.broadcast(dim=0, size=P_MAX)

    bias_sb = nl.ndarray(
        (P_MAX, cfg.h_tile.tile_size),
        dtype=bias_view.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=bias_sb[:P_MAX, :curr_h_block_size], src=broadcasted_bias_view.get_view())

    return bias_sb


def load_input_tensor_float(
    attention_view: TensorView,
    cfg: TilingConfig,
    target_dtype=None,
) -> List[nl.ndarray]:
    """
    Load input attention tensors for float (non-quantized) projection.

    Args:
        attention_view (TensorView): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        cfg (TilingConfig): Tiling configuration.
        target_dtype: Target dtype for tensor. If None, uses attention_view.dtype.

    Returns:
        List[nl.ndarray]: [n_size][d_size, s_block_size], Attention tensors in SBUF.
    """
    curr_s_tile_size = attention_view.shape[2]
    dtype = target_dtype if target_dtype != None else attention_view.dtype
    attention_sb = []

    for head_idx in range(cfg.n_size):
        attention_tensor = nl.ndarray(
            (cfg.d_size, cfg.s_tile.tile_size),
            dtype=dtype,
            buffer=nl.sbuf,
        )
        attn_head_view = attention_view.select(dim=0, index=head_idx)
        nisa.dma_copy(attention_tensor[: cfg.d_size, :curr_s_tile_size], attn_head_view.get_view())
        attention_sb.append(attention_tensor)

    return attention_sb


def load_input_tensor_quantized(
    attention_view: TensorView,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """
    Load input attention tensors for quantized projection (without quantization).

    Args:
        attention_view (TensorView): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.ndarray]: Attention tensors in SBUF (NOT quantized yet).
            - Double row: [n_size // 2][d_size, 2, s_block_size]
            - Normal: [n_size][d_size, s_block_size]
    """
    if not quant_config.use_double_row:
        return load_input_tensor_float(
            attention_view=attention_view,
            cfg=cfg,
        )

    curr_s_tile_size = attention_view.shape[2]
    num_heads_to_process = cfg.n_size // 2
    attention_sb = []

    for head_pair_idx in range(num_heads_to_process):
        attention_tensor = nl.ndarray(
            (cfg.d_size, 2, cfg.s_tile.tile_size),
            dtype=attention_view.dtype,
            buffer=nl.sbuf,
        )

        # First head
        attn_view_0 = attention_view.select(dim=0, index=head_pair_idx * 2)
        nisa.dma_copy(attention_tensor[: cfg.d_size, 0:1, :curr_s_tile_size], attn_view_0.get_view())

        # Second head
        attn_view_1 = attention_view.select(dim=0, index=head_pair_idx * 2 + 1)
        nisa.dma_copy(attention_tensor[: cfg.d_size, 1:2, :curr_s_tile_size], attn_view_1.get_view())

        attention_sb.append(attention_tensor)

    return attention_sb


def load_float_weights(
    weight_view: TensorView,
    cfg: TilingConfig,
    weight_dtype=nl.bfloat16,
) -> List[nl.ndarray]:
    """
    Load weights into SBUF for float (non-quantized) projection.

    Args:
        weight_view (TensorView): View of weight tensor [N, D, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.
        weight_dtype: Data type for weight tensor.

    Returns:
        List[nl.ndarray]: [n_size][d_size, h_block_size], Weight tensors in SBUF.

    Notes:
        - If curr_h_block_size < h_block_size, tensors contain garbage at end.
        - DMA operations only use valid elements.
    """
    curr_h_block_size = weight_view.shape[2]
    w_sbuf = []

    for head_idx in range(cfg.n_size):
        w_tensor = nl.ndarray(
            (cfg.d_size, cfg.h_tile.tile_size),
            dtype=weight_dtype,
            buffer=nl.sbuf,
        )
        head_weight_view = weight_view.select(dim=0, index=head_idx)
        nisa.dma_copy(w_tensor[:, :curr_h_block_size], head_weight_view.get_view())
        w_sbuf.append(w_tensor)

    return w_sbuf


def load_quantized_weights(
    weight_view: TensorView,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """
    Load quantized weights into SBUF for quantized projection.

    Supports both normal and double row formats based on quant_config.

    Args:
        weight_view (TensorView): View of weight tensor [N, D, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.ndarray]: Weight tensors in SBUF.
            - Normal: [n_size][d_size, h_block_size]
            - Double row: [n_size // 2][d_size, 2, h_block_size]

    Notes:
        - If curr_h_block_size < h_block_size, tensors contain garbage at end.
        - DMA operations only use valid elements.
    """
    if not quant_config.use_double_row:
        return load_float_weights(
            weight_view=weight_view,
            cfg=cfg,
            weight_dtype=quant_config.quant_data_type,
        )

    curr_h_block_size = weight_view.shape[2]
    w_sbuf = []

    for head_pair_idx in range(cfg.n_size // 2):
        # If curr_h_block_size < cfg.h_tile.tile_size, tensor will contain
        # some garbage data at the end. DMA operations only use valid elements.
        w_tensor = nl.ndarray(
            (cfg.d_size, 2, cfg.h_tile.tile_size),
            dtype=quant_config.quant_data_type,
            buffer=nl.sbuf,
        )

        # Load first head of the pair
        weight_view_0 = weight_view.select(dim=0, index=head_pair_idx * 2)
        nisa.dma_copy(dst=w_tensor[: cfg.d_size, 0:1, :curr_h_block_size], src=weight_view_0.get_view())

        # Load second head of the pair
        weight_view_1 = weight_view.select(dim=0, index=head_pair_idx * 2 + 1)
        nisa.dma_copy(dst=w_tensor[: cfg.d_size, 1:2, :curr_h_block_size], src=weight_view_1.get_view())

        w_sbuf.append(w_tensor)

    return w_sbuf


# ============================================================================
# MX Quantization Load Functions
# ============================================================================


def load_mx_scales_strided(data_p: int, scale_view: TensorView, padded_f: int = None) -> nl.ndarray:
    """Load MX scales from HBM and stride across partition-dim quadrants.

    Args:
        data_p: P dimension of the data tile (must be multiple of 32, <= 128).
        scale_view: View of scale tensor [data_p//_q_height, F].
        padded_f: Optional padded F dimension. If provided and > scale_f, allocates
                  larger buffer and zero-pads the extra columns.

    Returns:
        Scale tensor [data_p, F] (or [data_p, padded_f] if provided) in SBUF with strided layout.

    Notes:
        Scatter pattern places scales at positions 0-3, 32-35, 64-67, 96-99.
    """
    scale_p, scale_f = scale_view.shape
    out_f = padded_f if padded_f != None else scale_f

    if data_p > _SBUF_QUADRANT_SIZE:
        scale_sbuf = nl.ndarray((data_p, out_f), dtype=scale_view.dtype, buffer=nl.sbuf)
        nisa.memset(dst=scale_sbuf, value=0, engine=nisa.gpsimd_engine)
        for quadrant_idx in range(scale_p // _q_width):
            src_slice = scale_view.slice(dim=0, start=quadrant_idx * _q_width, end=quadrant_idx * _q_width + _q_width)
            nisa.dma_copy(
                src=src_slice.get_view(),
                dst=scale_sbuf[
                    quadrant_idx * _SBUF_QUADRANT_SIZE : quadrant_idx * _SBUF_QUADRANT_SIZE + _q_width, :scale_f
                ],
            )
    else:
        scale_sbuf = nl.ndarray((scale_p, out_f), dtype=scale_view.dtype, buffer=nl.sbuf)
        if padded_f != None and padded_f > scale_f:
            nisa.memset(dst=scale_sbuf[:, scale_f:out_f], value=0, engine=nisa.gpsimd_engine)
        nisa.dma_copy(src=scale_view.get_view(), dst=scale_sbuf[:, :scale_f])

    return scale_sbuf


def load_mx_weight_scales(
    weight_scale_view: TensorView,
    cfg: TilingConfig,
) -> List[nl.ndarray]:
    """Load and stride weight scales for all d_tiles.

    Args:
        weight_scale_view: View of weight scales [D // _q_height, curr_h_block_size].
        cfg: Tiling configuration.

    Returns:
        List of strided scale tensors per d_tile.
    """
    w_scale_sbuf_list = []
    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_size, actual_size = cfg.d_tile.get_bounds(d_tile_idx)
        scale_slice = weight_scale_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size // _q_height,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size // _q_height + actual_size // _q_height,
        )
        w_scale_sbuf = load_mx_scales_strided(padded_size, scale_slice)
        w_scale_sbuf_list.append(w_scale_sbuf)
    return w_scale_sbuf_list


def load_mx_quantized_weights(
    weight_view: TensorView,
    weight_dtype,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """Load pre-quantized MX weights for all d_tiles.

    Args:
        weight_view: View of weights [D, curr_h_block_size].
        cfg: Tiling configuration.
        quant_config: Quantization configuration.

    Returns:
        List of weight tensors per d_tile.

    Notes:
        Zero-pads last d_tile if padding is needed.
    """
    w_sbuf_list = []
    curr_h_block_size = weight_view.shape[1]
    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_size, actual_size = cfg.d_tile.get_bounds(d_tile_idx)
        if quant_config.is_mxfp8_static_quantized:
            w_sbuf = nl.ndarray((padded_size, cfg.h_tile.tile_size, _q_width), dtype=weight_dtype, buffer=nl.sbuf)
        else:
            w_sbuf = nl.ndarray((padded_size, cfg.h_tile.tile_size), dtype=weight_dtype, buffer=nl.sbuf)
        if cfg.d_tile.needs_padding(d_tile_idx):
            nisa.memset(dst=w_sbuf[actual_size:padded_size, :], value=0, engine=nisa.gpsimd_engine)
        w_slice = weight_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size + actual_size,
        ).get_view()
        if quant_config.is_mxfp8_static_quantized:
            nisa.dma_copy(dst=w_sbuf[:actual_size, :curr_h_block_size, :_q_width], src=w_slice)
            w_sbuf_list.append(w_sbuf.reshape((padded_size, cfg.h_tile.tile_size * _q_width)))
        else:
            nisa.dma_copy(dst=w_sbuf[:actual_size, :curr_h_block_size], src=w_slice)
            w_sbuf_list.append(w_sbuf)
    return w_sbuf_list


def load_mx_input_interleaved(
    attention_view: TensorView,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.ndarray]:
    """Load input attention and transpose for MX quantization.

    Args:
        attention_view: View of attention [D, _q_width, curr_s_tile_size].
        cfg: Tiling configuration.
        quant_config: Quantization configuration.

    Returns:
        List of transposed attention tensors [d_tile_count] -> [d_tile_size, padded_S, _q_width].

    Notes:
        Transposes [D, _q_width, S] -> [D, S, _q_width] for quantize_mx input format.
        Zero-pads D and S dimensions if padding is needed.

        Engine strategy adapts based on d_tile_count to maximize throughput:
        - d_tile_count == 1: Split work within tile between vector and scalar engines.
          Both engines process half the data in parallel, minimizing latency for single tile.
        - d_tile_count >= 2: Alternate engines across tiles (even tiles use vector, odd use scalar).
          This enables pipelining where load+quantize for d_tile N on one engine overlaps with
          load+quantize for d_tile N+1 on the other engine, since quantization depends on
          the same d_tile's load completing.
    """
    curr_s_tile_size = attention_view.shape[2]
    # nc_matmul_mx requires even number of elements in free dimension
    padded_s_tile_size = curr_s_tile_size + (curr_s_tile_size % 2)
    attn_sbuf_list = []
    use_pipeline_strategy = cfg.d_tile.tile_info.tile_count >= 2

    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_d_size, actual_d_size = cfg.d_tile.get_bounds(d_tile_idx)
        if padded_d_size <= 0:
            break

        tmp_sbuf = nl.ndarray(
            (padded_d_size, _q_width, padded_s_tile_size),
            dtype=quant_config.input_data_type,
            buffer=nl.sbuf,
        )
        inp_sbuf = nl.ndarray(
            (padded_d_size, padded_s_tile_size, _q_width),
            dtype=quant_config.input_data_type,
            buffer=nl.sbuf,
        )
        if cfg.d_tile.needs_padding(d_tile_idx):
            nisa.memset(dst=tmp_sbuf[actual_d_size:padded_d_size, :, :], value=0, engine=nisa.gpsimd_engine)
        if curr_s_tile_size % 2 == 1:
            nisa.memset(dst=tmp_sbuf[:, :, curr_s_tile_size:padded_s_tile_size], value=0, engine=nisa.gpsimd_engine)
        attn_slice = attention_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size + actual_d_size,
        )

        nisa.dma_copy(src=attn_slice.get_view(), dst=tmp_sbuf[:actual_d_size, :, :curr_s_tile_size])

        tmp_view = TensorView(tmp_sbuf).permute([0, 2, 1])
        if use_pipeline_strategy:
            # Pipeline across d_tiles: each tile uses one engine, enabling overlap with
            # subsequent quantization on the same engine while other d_tiles use the other engine
            engine = nisa.vector_engine if d_tile_idx % 2 == 0 else nisa.scalar_engine
            nisa.tensor_copy(src=tmp_view.get_view(), dst=inp_sbuf, engine=engine)
        else:
            # Single d_tile: split work between engines for parallelism within tile
            nisa.tensor_copy(
                src=tmp_view.slice(dim=2, start=0, end=_q_width // 2).get_view(),
                dst=inp_sbuf[:, :, : _q_width // 2],
                engine=nisa.vector_engine,
            )
            nisa.tensor_copy(
                src=tmp_view.slice(dim=2, start=_q_width // 2, end=_q_width).get_view(),
                dst=inp_sbuf[:, :, _q_width // 2 :],
                engine=nisa.scalar_engine,
            )
        attn_sbuf_list.append(inp_sbuf)
    return attn_sbuf_list, padded_s_tile_size


def load_mx_prequantized_input(
    attention_view: TensorView,
    input_scale_view: TensorView,
    cfg: TilingConfig,
) -> Tuple[List[nl.ndarray], List[nl.ndarray], int]:
    """Load pre-quantized MX input attention and scales for all d_tiles.

    Used when input is already quantized to float8_e4m3fn_x4 format with pre-computed
    scales, skipping the on-device quantization step.

    Args:
        attention_view: View of pre-quantized attention [D, curr_s_tile_size] (float8_e4m3fn_x4).
        input_scale_view: View of input scales [D//_q_height, curr_s_tile_size] (uint8).
        cfg: Tiling configuration.

    Returns:
        Tuple of (quant_attn_list, attn_scale_list, padded_s_tile_size):
            - quant_attn_list: [d_tile_count] -> [d_tile_size, s_tile_size] quantized attention
            - attn_scale_list: [d_tile_count] -> [d_tile_size, s_tile_size] strided scales
            - padded_s_tile_size: S tile size padded to even number

    Notes:
        - Reuses load_mx_scales_strided for scale loading with proper stride pattern
        - Zero-pads D and S dimensions if padding is needed for nc_matmul_mx
    """
    curr_s_tile_size = attention_view.shape[1]
    # nc_matmul_mx requires even number of elements in free dimension
    padded_s_tile_size = curr_s_tile_size + (curr_s_tile_size % 2)

    quant_attn_list = []
    attn_scale_list = []

    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_d_size, actual_d_size = cfg.d_tile.get_bounds(d_tile_idx)
        if padded_d_size <= 0:
            break

        # Load quantized attention (similar to load_mx_quantized_weights)
        quant_attn_sbuf = nl.ndarray((padded_d_size, cfg.s_tile.tile_size), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        if cfg.d_tile.needs_padding(d_tile_idx):
            nisa.memset(dst=quant_attn_sbuf[actual_d_size:padded_d_size, :], value=0, engine=nisa.gpsimd_engine)
        if curr_s_tile_size % 2 == 1:
            nisa.memset(dst=quant_attn_sbuf[:, curr_s_tile_size:padded_s_tile_size], value=0, engine=nisa.gpsimd_engine)

        attn_slice = attention_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size + actual_d_size,
        )
        nisa.dma_copy(dst=quant_attn_sbuf[:actual_d_size, :curr_s_tile_size], src=attn_slice.get_view())
        quant_attn_list.append(quant_attn_sbuf)

        # Load input scales using strided pattern
        scale_slice = input_scale_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size // _q_height,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size // _q_height + actual_d_size // _q_height,
        )
        attn_scale_sbuf = load_mx_scales_strided(padded_d_size, scale_slice, padded_s_tile_size)
        attn_scale_list.append(attn_scale_sbuf)

    return quant_attn_list, attn_scale_list, padded_s_tile_size


def create_constant_mx_scales(p_size: int, f_size: int, scale_value: int = 127) -> nl.ndarray:
    """
    Create constant MX scale tensor for static MX quantization.

    For static MX FP8, we use a constant scale value (126) so that nc_matmul_mx
    dequantizes as 2^(127-127) = 1. The actual dequantization is done separately
    using the static scales.

    Args:
        p_size (int): Partition dimension size.
        f_size (int): Free dimension size.
        scale_value (int): Constant scale value (default 127).

    Returns:
        nl.ndarray: [p_size, f_size], Constant scale tensor in SBUF.
    """
    scale_sbuf = nl.ndarray((p_size, f_size), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=scale_sbuf, value=scale_value, engine=nisa.gpsimd_engine)
    return scale_sbuf
