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

from typing import List

import nki.isa as nisa
import nki.language as nl

from ...utils.tensor_view import TensorView
from .output_projection_cte_parameters import QuantizationConfig, TilingConfig


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
    nisa.memset(bias_vector_sbuf, value=0.0)
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
    dtype = target_dtype if target_dtype is not None else attention_view.dtype
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
