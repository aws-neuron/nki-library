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

from .output_projection_cte_parameters import _SBUF_QUADRANT_SIZE, QuantizationConfig, TilingConfig, _q_height, _q_width


def get_zero_bias_vector_sbuf(tile_size: int, activation_data_type=nl.float32) -> nl.NkiTensor:
    """
    Create zero bias vector in SBUF.

    Args:
        tile_size (int): Size of the bias vector.
        activation_data_type: Data type for the bias vector.

    Returns:
        nl.NkiTensor: [tile_size, 1], Zero-initialized bias vector in SBUF.
    """
    bias_vector_sbuf = nl.ndarray((tile_size, 1), dtype=activation_data_type, buffer=nl.sbuf)
    nisa.memset(bias_vector_sbuf, value=0.0, engine=nisa.gpsimd_engine)
    return bias_vector_sbuf


def invert_static_quant_scales(scales_sbuf: nl.NkiTensor) -> None:
    """
    Invert scales for quantization (reciprocal).

    Args:
        scales_sbuf (nl.NkiTensor): Scales tensor in SBUF to invert in-place.

    Returns:
        None: Modifies scales_sbuf in-place.
    """
    nisa.reciprocal(scales_sbuf, scales_sbuf)


def load_static_quant_input_scales(static_quant_scale_hbm: nl.NkiTensor) -> nl.NkiTensor:
    """
    Load and prepare input quantization scales.

    Args:
        static_quant_scale_hbm (nl.NkiTensor): [P_MAX, 1], Input scales tensor in HBM.

    Returns:
        nl.NkiTensor: [P_MAX, 1], Input scales in SBUF.
    """
    P_MAX = nl.tile_size.pmax
    static_quant_scale_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    # Layout: [P_MAX, 1] where P_MAX=128 partition elements, 1 free dimension
    scale_view = static_quant_scale_hbm.slice(dim=0, start=0, end=P_MAX)
    nisa.dma_copy(dst=static_quant_scale_sbuf, src=scale_view)
    return static_quant_scale_sbuf


def load_static_quant_weight_scales(
    static_quant_weight_scale_hbm: nl.NkiTensor,
    static_quant_input_scale_sbuf: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    Load weight scales and multiply with input scales for dequantization.

    Args:
        static_quant_weight_scale_hbm (nl.NkiTensor): [P_MAX, 1], Weight scales tensor in HBM.
        static_quant_input_scale_sbuf (nl.NkiTensor): [P_MAX, 1], Input scales in SBUF.

    Returns:
        nl.NkiTensor: [P_MAX, 1], Combined weight scales in SBUF (weight_scale * input_scale).
    """
    P_MAX = nl.tile_size.pmax
    bias_vector = get_zero_bias_vector_sbuf(P_MAX)
    static_quant_weight_scale_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Layout: [P_MAX, 1] where P_MAX=128 partition elements, 1 free dimension
    weight_scale_view = static_quant_weight_scale_hbm.slice(dim=0, start=0, end=P_MAX)
    nisa.dma_copy(dst=static_quant_weight_scale_sbuf, src=weight_scale_view)

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
    bias_view: nl.NkiTensor,
    cfg: TilingConfig,
) -> nl.NkiTensor:
    """
    Load bias into SBUF and broadcast to [P_MAX, h_block_size].

    Uses broadcast on the HBM tensor and loads directly into the
    broadcasted array for better DMA utilization.

    Args:
        bias_view (NkiTensor): View of bias tensor [1, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.

    Returns:
        nl.NkiTensor: [P_MAX, h_block_size], Broadcasted bias tensor in SBUF.

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
    nisa.dma_copy(dst=bias_sb[:P_MAX, :curr_h_block_size], src=broadcasted_bias_view)

    return bias_sb


def load_input_tensor_float(
    attention_view: nl.NkiTensor,
    cfg: TilingConfig,
    target_dtype=None,
) -> List[nl.NkiTensor]:
    """
    Load input attention tensors for float (non-quantized) projection.

    Args:
        attention_view (NkiTensor): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        cfg (TilingConfig): Tiling configuration.
        target_dtype: Target dtype for tensor. If None, uses attention_view.dtype.

    Returns:
        List[nl.NkiTensor]: [n_size][d_size, s_block_size], Attention tensors in SBUF.
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
        nisa.dma_copy(attention_tensor[: cfg.d_size, :curr_s_tile_size], attn_head_view)
        attention_sb.append(attention_tensor)

    return attention_sb


def load_input_tensor_float_transposed(
    attention_view: nl.NkiTensor,
    cfg: TilingConfig,
    target_dtype=None,
) -> List[nl.NkiTensor]:
    """
    Load attention for the float path from the untransposed [S, N, D] layout, producing
    the same D-on-partition SBUF tiles as ``load_input_tensor_float``.

    N and D are adjacent here, so the view flattens to [S, N*D] and each packed head is one
    contiguous [S, d_size] column band, loaded by a single transpose.

    Args:
        attention_view (NkiTensor): View of attention for current batch/s_block
            [curr_s_tile_size, n_size, d_size].
        cfg (TilingConfig): Tiling configuration.
        target_dtype: Target dtype for tensor. If None, uses attention_view.dtype.

    Returns:
        List[nl.NkiTensor]: [n_size][d_size, s_block_size], Attention tensors in SBUF.
    """
    curr_s_tile_size = attention_view.shape[0]
    dtype = target_dtype if target_dtype != None else attention_view.dtype
    attention_sb = []

    # [S, N, D] -> [S, N*D]: heads become contiguous column bands of width d_size.
    flat_view = attention_view.reshape((curr_s_tile_size, cfg.n_size * cfg.d_size))

    for head_idx in range(cfg.n_size):
        attention_tensor = nl.ndarray(
            (cfg.d_size, cfg.s_tile.tile_size),
            dtype=dtype,
            buffer=nl.sbuf,
        )
        d_start = head_idx * cfg.d_size
        head_view = flat_view.slice(dim=1, start=d_start, end=d_start + cfg.d_size)
        # One transpose covers the whole S tile: the <=128 limit is on src's minor axis (D),
        # and S lands on dst's free axis, which is unconstrained.
        nisa.dma_transpose(dst=attention_tensor[: cfg.d_size, :curr_s_tile_size], src=head_view)
        attention_sb.append(attention_tensor)

    return attention_sb


def _is_legal_transpose_start_partition(start_partition: int, num_partitions: int) -> bool:
    """
    Whether a transpose DMA may write ``num_partitions`` rows starting at ``start_partition``.

    Transpose DMA inherits the SBUF start-partition rule: a tensor access spanning
    ``num_partitions`` partitions must begin on a quadrant/half boundary wide enough to hold
    it. Violating it fails compilation with ``NCC_IDMA007`` (illegal starting partition).

    Args:
        start_partition (int): First partition the access writes.
        num_partitions (int): Number of partitions the access spans.

    Returns:
        bool: True when the access is legal.
    """
    if num_partitions > 64:
        return start_partition == 0
    if num_partitions > 32:
        return start_partition % 64 == 0
    return start_partition % _SBUF_QUADRANT_SIZE == 0


def load_input_tensor_float_heads_outer(
    attention_view: nl.NkiTensor,
    cfg: TilingConfig,
    target_dtype=None,
) -> List[nl.NkiTensor]:
    """
    Load attention for the float path from the untransposed heads-outer [N, S, D] layout,
    producing the same D-on-partition SBUF tiles as ``load_input_tensor_float``.

    Unlike [S, N, D], N and D are not adjacent here, so head packing cannot be a reshape of
    the HBM view — a packed tile is assembled from up to ``group_size`` separate head slices.
    Each packed tile's partition rows map onto the flattened N*D axis at
    ``packed_head_idx * d_size + row``, which resolves to original head ``flat // d_orig``
    at column offset ``flat % d_orig``. That covers both packing directions: N folded into D
    (``d_size`` spans whole heads) and D folded back into N (``d_size`` is a column band of
    one head).

    A band starting at a partition the transpose DMA cannot address directly (see
    ``_is_legal_transpose_start_partition``) is staged through a scratch tile based at
    partition 0 and then copied into place, which a plain DMA can do at any offset.

    Args:
        attention_view (NkiTensor): View of attention for current batch/s_block
            [n_orig, curr_s_tile_size, d_orig].
        cfg (TilingConfig): Tiling configuration.
        target_dtype: Target dtype for tensor. If None, uses attention_view.dtype.

    Returns:
        List[nl.NkiTensor]: [n_size][d_size, s_block_size], Attention tensors in SBUF.
    """
    curr_s_tile_size = attention_view.shape[1]
    d_orig = attention_view.shape[2]
    dtype = target_dtype if target_dtype != None else attention_view.dtype
    attention_sb = []

    for packed_head_idx in range(cfg.n_size):
        attention_tensor = nl.ndarray(
            (cfg.d_size, cfg.s_tile.tile_size),
            dtype=dtype,
            buffer=nl.sbuf,
        )
        # Walk the packed tile's partition rows, one original-head column band at a time.
        row = 0
        while row < cfg.d_size:
            flat_idx = packed_head_idx * cfg.d_size + row
            head_idx = flat_idx // d_orig
            d_offset = flat_idx % d_orig
            chunk = min(d_orig - d_offset, cfg.d_size - row)
            head_view = attention_view.select(dim=0, index=head_idx).slice(dim=1, start=d_offset, end=d_offset + chunk)
            if _is_legal_transpose_start_partition(row, chunk):
                dst_tile = attention_tensor
                dst_row = row
            else:
                # Transpose into a partition-0 scratch tile, then relocate with a plain copy.
                dst_tile = nl.ndarray((chunk, cfg.s_tile.tile_size), dtype=dtype, buffer=nl.sbuf)
                dst_row = 0

            nisa.dma_transpose(
                dst=dst_tile[dst_row : dst_row + chunk, :curr_s_tile_size],
                src=head_view,
            )
            if dst_tile is not attention_tensor:
                nisa.dma_copy(
                    dst=attention_tensor[row : row + chunk, :curr_s_tile_size],
                    src=dst_tile[:chunk, :curr_s_tile_size],
                )
            row += chunk
        attention_sb.append(attention_tensor)

    return attention_sb


def load_input_tensor_quantized(
    attention_view: nl.NkiTensor,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """
    Load input attention tensors for quantized projection (without quantization).

    Args:
        attention_view (NkiTensor): View of attention tensor for current batch/s_block [N, D, curr_s_tile_size].
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Attention tensors in SBUF (NOT quantized yet).
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
        nisa.dma_copy(attention_tensor[: cfg.d_size, 0:1, :curr_s_tile_size], attn_view_0)

        # Second head
        attn_view_1 = attention_view.select(dim=0, index=head_pair_idx * 2 + 1)
        nisa.dma_copy(attention_tensor[: cfg.d_size, 1:2, :curr_s_tile_size], attn_view_1)

        attention_sb.append(attention_tensor)

    return attention_sb


def load_float_weights(
    weight_view: nl.NkiTensor,
    cfg: TilingConfig,
    weight_dtype=nl.bfloat16,
) -> List[nl.NkiTensor]:
    """
    Load weights into SBUF for float (non-quantized) projection.

    Args:
        weight_view (NkiTensor): View of weight tensor [N, D, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.
        weight_dtype: Data type for weight tensor.

    Returns:
        List[nl.NkiTensor]: [n_size][d_size, h_block_size], Weight tensors in SBUF.

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
        nisa.dma_copy(w_tensor[:, :curr_h_block_size], head_weight_view)
        w_sbuf.append(w_tensor)

    return w_sbuf


def load_quantized_weights(
    weight_view: nl.NkiTensor,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
    """
    Load quantized weights into SBUF for quantized projection.

    Supports both normal and double row formats based on quant_config.

    Args:
        weight_view (NkiTensor): View of weight tensor [N, D, curr_h_block_size] for current h_block.
        cfg (TilingConfig): Tiling configuration.
        quant_config (QuantizationConfig): Quantization configuration.

    Returns:
        List[nl.NkiTensor]: Weight tensors in SBUF.
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
        nisa.dma_copy(dst=w_tensor[: cfg.d_size, 0:1, :curr_h_block_size], src=weight_view_0)

        # Load second head of the pair
        weight_view_1 = weight_view.select(dim=0, index=head_pair_idx * 2 + 1)
        nisa.dma_copy(dst=w_tensor[: cfg.d_size, 1:2, :curr_h_block_size], src=weight_view_1)

        w_sbuf.append(w_tensor)

    return w_sbuf


# ============================================================================
# MX Quantization Load Functions
# ============================================================================


def load_mx_scales_strided(data_p: int, scale_view: nl.NkiTensor, padded_f: int = None) -> nl.NkiTensor:
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
                src=src_slice,
                dst=scale_sbuf[
                    quadrant_idx * _SBUF_QUADRANT_SIZE : quadrant_idx * _SBUF_QUADRANT_SIZE + _q_width, :scale_f
                ],
            )
    else:
        scale_sbuf = nl.ndarray((scale_p, out_f), dtype=scale_view.dtype, buffer=nl.sbuf)
        if padded_f != None and padded_f > scale_f:
            nisa.memset(dst=scale_sbuf[:, scale_f:out_f], value=0, engine=nisa.gpsimd_engine)
        nisa.dma_copy(src=scale_view, dst=scale_sbuf[:, :scale_f])

    return scale_sbuf


def load_mx_weight_scales(
    weight_scale_view: nl.NkiTensor,
    cfg: TilingConfig,
) -> List[nl.NkiTensor]:
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


def load_mx_compact_weight_scales(
    weight_scale_hbm: nl.NkiTensor,
    h_start: int,
    curr_h_block_size: int,
    cfg: TilingConfig,
) -> List[nl.NkiTensor]:
    """Load block-128 compact MX weight scales and expand to hardware layout.

    quantizes weights with one uint8 scale per 128x128 block, so the
    HBM scale tensor is ``[K_unpacked // 128, H // 128]`` — far more compact than
    the standard MX block-32 layout ``[K_unpacked // 8, H]``. The hardware
    ``nc_matmul_mx`` op consumes the dense MX scale layout
    ``[d_tile_size, h_tile_size]`` with sparse quadrant population (rows
    ``0-3, 32-35, 64-67, 96-99`` carry valid scales).

    Each d_tile of ``d_tile_size`` partitions covers ``d_tile_size * 4`` actual
    K-elements (since the data is fp8x4 packed, 4 K-elements per partition byte).
    Within a d_tile, the 4 hardware quadrants of 32 partitions each cover a
    distinct 128-K-element region, so each quadrant maps to a different compact
    scale row. Mirroring ``qkv_mla_cte_utils._load_mx_weights``:

    1. DMA stage: per quadrant, copy that quadrant's compact row into the 4
       valid partition rows. Stride-0 on the partition axis broadcasts within
       a quadrant; stride 1 on the free dim reads ``out_n_blocks`` consecutive
       compact columns. Free-dim broadcast at DMA time is NOT used — the DMA
       engine doesn't support stride-0 on the free axis without falling back
       to per-element copies.
    2. Compute stage: vector-engine ``tensor_copy`` with stride-0 source
       broadcast on the free axis expands each compact column to ``SCALE_BLOCK``
       (128) hardware columns, producing the
       ``[d_tile_size, n_blocks * SCALE_BLOCK]`` layout consumed by
       ``nc_matmul_mx``.

    Args:
        weight_scale_hbm: ``[K_unpacked_full // 128, H_full // 128]`` uint8
            compact block-128 scales on HBM.
        h_start: Column offset (in H elements) of the current h_block. Must be
            a multiple of 128.
        curr_h_block_size: Width of the current h_block in H elements.
        cfg: Tiling configuration. ``cfg.d_tile`` describes the D-tiling.

    Returns:
        List of dense MX scale tensors per d_tile, each
        ``[d_tile_size, padded_h_block_size]`` uint8 in hardware layout.
    """
    SCALE_BLOCK = 128
    SCALE_P_PER_QUAD = 4
    H_PACK = 4

    # Slice scale to current h_block. ceil(curr_h_block_size / 128) compact columns.
    n_block_offset = h_start // SCALE_BLOCK
    out_n_blocks = (curr_h_block_size + SCALE_BLOCK - 1) // SCALE_BLOCK
    padded_h_block_size = out_n_blocks * SCALE_BLOCK
    full_n_blocks = weight_scale_hbm.shape[1]

    compact_rows_per_dtile = (cfg.d_tile.tile_info.tile_size * H_PACK) // SCALE_BLOCK

    w_scale_sbuf_list = []
    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_size, actual_size = cfg.d_tile.get_bounds(d_tile_idx)
        if padded_size <= 0:
            break

        # Final destination: dense MX layout for nc_matmul_mx.
        scale_sbuf = nl.ndarray((padded_size, padded_h_block_size), dtype=nl.uint8, buffer=nl.sbuf)

        # 4 rows per quadrant carry the per-quadrant compact
        # scale row, broadcast 4-wide on partition.
        compact_sb = nl.ndarray((padded_size, out_n_blocks), dtype=nl.uint8, buffer=nl.sbuf)

        compact_base_row = d_tile_idx * compact_rows_per_dtile

        # Number of quadrants in this d_tile that map to valid (non-padding) compact rows.
        valid_k_elements = actual_size * H_PACK
        valid_quadrants = valid_k_elements // SCALE_BLOCK
        num_quadrants = padded_size // _SBUF_QUADRANT_SIZE

        # Stage 1: DMA per quadrant with stride-0 partition broadcast (4 rows).
        # Each quadrant reads a distinct compact row covering its 128-K slice.
        for quad_idx in range(min(num_quadrants, valid_quadrants)):
            nisa.dma_copy(
                dst=compact_sb[
                    quad_idx * _SBUF_QUADRANT_SIZE : quad_idx * _SBUF_QUADRANT_SIZE + SCALE_P_PER_QUAD,
                    :out_n_blocks,
                ],
                src=weight_scale_hbm[
                    compact_base_row + quad_idx : compact_base_row + quad_idx + 1,
                    n_block_offset : n_block_offset + out_n_blocks,
                ].broadcast(0, SCALE_P_PER_QUAD),
            )

        # Stage 2: vector-engine broadcast on free dim (stride-0 source on inner dim).
        # Treat compact_sb as [P, n_blocks, 1] and broadcast inner dim by SCALE_BLOCK.
        src_view = compact_sb.expand_dim(dim=2).broadcast(dim=2, size=SCALE_BLOCK)
        dst_view = scale_sbuf.reshape_dim(dim=1, shape=(out_n_blocks, SCALE_BLOCK))
        nisa.tensor_copy(dst=dst_view, src=src_view)

        w_scale_sbuf_list.append(scale_sbuf)
    return w_scale_sbuf_list


def load_mx_quantized_weights(
    weight_view: nl.NkiTensor,
    weight_dtype,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
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
    # HBM weights may carry a same-width container label (uint32 mxfp8 / uint16 mxfp4); DMA needs
    # matching element widths, so allocate SBUF at the HBM dtype and view-cast to the MX dtype after.
    hbm_weight_dtype = weight_view.dtype
    for d_tile_idx in range(cfg.d_tile.tile_info.tile_count):
        padded_size, actual_size = cfg.d_tile.get_bounds(d_tile_idx)
        if quant_config.is_mxfp8_static_quantized or quant_config.is_row_mxfp8_quantized:
            w_sbuf = nl.ndarray((padded_size, cfg.h_tile.tile_size, _q_width), dtype=weight_dtype, buffer=nl.sbuf)
        else:
            w_sbuf = nl.ndarray((padded_size, cfg.h_tile.tile_size), dtype=hbm_weight_dtype, buffer=nl.sbuf)
        if cfg.d_tile.needs_padding(d_tile_idx):
            nisa.memset(dst=w_sbuf[actual_size:padded_size, :], value=0, engine=nisa.gpsimd_engine)
        w_slice = weight_view.slice(
            dim=0,
            start=d_tile_idx * cfg.d_tile.tile_info.tile_size,
            end=d_tile_idx * cfg.d_tile.tile_info.tile_size + actual_size,
        )
        if quant_config.is_mxfp8_static_quantized or quant_config.is_row_mxfp8_quantized:
            nisa.dma_copy(dst=w_sbuf[:actual_size, :curr_h_block_size, :_q_width], src=w_slice)
            w_sbuf_list.append(w_sbuf.reshape((padded_size, cfg.h_tile.tile_size * _q_width)))
        else:
            nisa.dma_copy(dst=w_sbuf[:actual_size, :curr_h_block_size], src=w_slice)
            if hbm_weight_dtype != weight_dtype:
                w_sbuf = w_sbuf.view(weight_dtype)
            w_sbuf_list.append(w_sbuf)
    return w_sbuf_list


def load_mx_input_interleaved(
    attention_view: nl.NkiTensor,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> List[nl.NkiTensor]:
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

        nisa.dma_copy(src=attn_slice, dst=tmp_sbuf[:actual_d_size, :, :curr_s_tile_size])

        tmp_view = tmp_sbuf.permute([0, 2, 1])
        if use_pipeline_strategy:
            # Pipeline across d_tiles: each tile uses one engine, enabling overlap with
            # subsequent quantization on the same engine while other d_tiles use the other engine
            engine = nisa.vector_engine if d_tile_idx % 2 == 0 else nisa.scalar_engine
            nisa.tensor_copy(src=tmp_view, dst=inp_sbuf, engine=engine)
        else:
            # Single d_tile: split work between engines for parallelism within tile
            nisa.tensor_copy(
                src=tmp_view.slice(dim=2, start=0, end=_q_width // 2),
                dst=inp_sbuf[:, :, : _q_width // 2],
                engine=nisa.vector_engine,
            )
            nisa.tensor_copy(
                src=tmp_view.slice(dim=2, start=_q_width // 2, end=_q_width),
                dst=inp_sbuf[:, :, _q_width // 2 :],
                engine=nisa.scalar_engine,
            )
        attn_sbuf_list.append(inp_sbuf)
    return attn_sbuf_list, padded_s_tile_size


def load_mx_prequantized_input(
    attention_view: nl.NkiTensor,
    input_scale_view: nl.NkiTensor,
    cfg: TilingConfig,
) -> Tuple[List[nl.NkiTensor], List[nl.NkiTensor], int]:
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
        nisa.dma_copy(dst=quant_attn_sbuf[:actual_d_size, :curr_s_tile_size], src=attn_slice)
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


def create_constant_mx_scales(p_size: int, f_size: int, scale_value: int = 127) -> nl.NkiTensor:
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
        nl.NkiTensor: [p_size, f_size], Constant scale tensor in SBUF.
    """
    scale_sbuf = nl.ndarray((p_size, f_size), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=scale_sbuf, value=scale_value, engine=nisa.gpsimd_engine)
    return scale_sbuf


def load_row_weight_dequant_scales(
    weight_scale_hbm: nl.NkiTensor,
    h_start: int,
    curr_h_block_size: int,
    h_tile_size: int,
) -> nl.NkiTensor:
    """Load per-row weight dequant scales for ROW_MX quantization.

    Args:
        weight_scale_hbm: [P_MAX, H], Per-row weight dequant scales in HBM (pre-broadcast).
        h_start: Start offset in H dimension for current h_block.
        curr_h_block_size: Current H block size.
        h_tile_size: Allocated H tile size.

    Returns:
        nl.NkiTensor: [P_MAX, h_tile_size], Weight dequant scales in SBUF.
    """
    P_MAX = nl.tile_size.pmax
    scale_sbuf = nl.ndarray((P_MAX, h_tile_size), dtype=nl.float32, buffer=nl.sbuf)
    scale_view = weight_scale_hbm.slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
    nisa.dma_copy(dst=scale_sbuf[:P_MAX, :curr_h_block_size], src=scale_view)
    return scale_sbuf
