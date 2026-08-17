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
Output Projection TKG Primitives with Manual Allocation Support.

This version uses BufferManager (sbm) for explicit SBUF allocation control,
enabling scope-based memory management and multi-buffering optimizations.
"""

from typing import Optional

import nki.language as nl
from nki.language import affine_range

from ...core.output_projection.output_projection_utils import calculate_head_packing
from ...core.utils.allocator import BufferManager, create_auto_alloc_manager
from ...core.utils.common_types import QuantizationType
from ...core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_program_sharding_info
from ...core.utils.tiled_range import TiledRange
from ..primitives import ColMajor, RowMajor, blas, dma, tile_stream
from ..primitives.utils import max_tile
from ..primitives.view_spec import view
from .output_projection_tkg_utils import budget_weight_blocks

P_MAX = 128
F_MAX = 512
NUM_PSUM_BANKS = 8


def output_projection_primitives(
    attention: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.NkiTensor] = None,
    input_scale: Optional[nl.NkiTensor] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm: Optional[BufferManager] = None,
) -> nl.NkiTensor:
    # Create local sbm if not provided
    if sbm is None:
        sbm = create_auto_alloc_manager()

    sbm.open_scope(name="output_projection_tkg_primitives")

    if TRANSPOSE_OUT:
        result = _output_projection_transposed(
            attention, weight, bias, quantization_type, weight_scale, input_scale, sbm
        )
    else:
        result = _output_projection_regular(attention, weight, bias, quantization_type, weight_scale, input_scale, sbm)

    sbm.close_scope()
    return result


def _load_static_scales(weight_scale, input_scale, sbm):
    input_scale_buf = tile_stream.alloc_logical((P_MAX, 1), P_MAX, nl.float32, "in_scale", sbm=sbm)
    dma.load(input_scale_buf, input_scale.slice(0, 0, P_MAX))
    dequant_scale_buf = tile_stream.alloc_logical((P_MAX, 1), P_MAX, nl.float32, "dequant_scale", sbm=sbm)
    dma.load(dequant_scale_buf, weight_scale.slice(0, 0, P_MAX))
    blas.activation(dequant_scale_buf, op=nl.copy, scale=input_scale_buf)
    return input_scale_buf, dequant_scale_buf


def _load_and_shuffle_attn(
    attention: nl.NkiTensor,
    input_scale_buf: Optional[nl.NkiTensor],
    quantization_type: QuantizationType,
    d_packed: int,
    use_double_row: bool,
    sbm: BufferManager,
) -> nl.NkiTensor:
    d, b, n, s = attention.shape
    bxs = b * s
    is_static = quantization_type == QuantizationType.STATIC

    attn_loaded = tile_stream.alloc_logical((d, b, n, s), d, attention.dtype, "attn_loaded", sbm=sbm)
    dma.load(attn_loaded, attention)

    attn_dtype = nl.float8_e4m3 if is_static else attention.dtype

    attn_quantized = None
    if is_static:
        blas.reciprocal(input_scale_buf)
        blas.activation(attn_loaded, op=nl.copy, scale=input_scale_buf.slice(0, 0, d))
        max_pos_val = get_max_positive_value_for_dtype(attn_dtype)
        attn_quantized = tile_stream.alloc_logical((d, b, n, s), d, attn_dtype, "attn_quantized", sbm=sbm)
        blas.tensor_scalar(
            attn_quantized, attn_loaded, op0=nl.minimum, operand0=max_pos_val, op1=nl.maximum, operand1=-max_pos_val
        )

    attn_shape = (n * d // 2, 2, bxs) if use_double_row else (n * d, bxs)
    attn_buf = tile_stream.alloc_logical(attn_shape, d_packed, attn_dtype, "attn", sbm=sbm)

    # Shuffle attention to output layout
    shuffle_src = attn_quantized if is_static else attn_loaded
    tile_dims = (0, 2) if use_double_row else (0, 1)
    dma.TensorCopy(
        tile_stream.tile(attn_buf, (d, bxs), tile_dims=tile_dims, iter_order=ColMajor()),
        tile_stream.tile(shuffle_src, (d, b, s), tile_dims=(0, 1, 3), iter_order=RowMajor()),
    ).execute()

    return attn_buf


def _output_projection_regular(attention, weight, bias, quantization_type, weight_scale, input_scale, sbm):
    """Output: [B*S, H]"""
    d, b, n, s = attention.shape
    _, h = weight.shape
    bxs = b * s
    bxs_pmax = min(bxs, P_MAX)

    _, lnc, lnc_id = get_program_sharding_info()
    h_sharded = h // lnc
    h_start = lnc_id * h_sharded

    # Head packing: pack multiple heads into partition dimension
    if d % 32 == 0:
        n_packed, d_packed, group_size = calculate_head_packing(n, d, P_MAX)
    else:
        n_packed, d_packed, group_size = n, d, 1
    is_static = quantization_type == QuantizationType.STATIC
    is_row = quantization_type == QuantizationType.ROW

    # Double row optimization: STATIC only, even packed heads, N_packed*B divisible by 32, B*S >= 64
    use_double_row = is_static and n_packed % 2 == 0 and n_packed * b % 32 == 0 and bxs >= 64

    output = nl.ndarray((bxs, h), dtype=attention.dtype, buffer=nl.shared_hbm)

    # Load scales and compute combined_scale before shuffle
    input_scale_buf, dequant_scale_buf = (
        _load_static_scales(weight_scale, input_scale, sbm) if is_static else (None, None)
    )

    # Load, quantize (if STATIC), and shuffle attention
    attn_buf = _load_and_shuffle_attn(attention, input_scale_buf, quantization_type, d_packed, use_double_row, sbm)

    # Budget h_block_size, weight buffer slots, and output interleave degree
    # h_block_size is calculated to ensure at least 2 weight buffers fit (for double-buffering)
    num_bxs_tiles = div_ceil(bxs, P_MAX)

    h_block_size, num_w_h_blocks, out_sb_interleave_degree = budget_weight_blocks(
        h_sharded=h_sharded,
        num_bxs_tiles=num_bxs_tiles,
        w_dtype=weight.dtype,
        io_dtype=attention.dtype,
        n_size=n_packed,
        use_double_row=use_double_row,
        out_in_sb=False,
        sbm=sbm,
    )

    # Allocate circular buffer slots for weights
    wght_bufs = []
    wght_buf_shape = (n // 2 * d, 2, h_block_size) if use_double_row else (n * d, h_block_size)
    for w_idx in range(num_w_h_blocks):
        wght_bufs.append(tile_stream.alloc_logical(wght_buf_shape, d_packed, weight.dtype, f"wght_{w_idx}", sbm=sbm))

    wgt_hbm_grid = tile_stream.tile(weight.slice(1, h_start, h_start + h_sharded), (n * d, h_block_size))
    num_h_blocks = wgt_hbm_grid.get_num_tiles()

    # Preload all weights if they fit
    wgt_tile_dims = (0, 2) if use_double_row else None
    all_weights_preloaded = num_w_h_blocks == num_h_blocks
    if all_weights_preloaded:
        for h_i in affine_range(num_h_blocks):
            wgt_hbm_tile = wgt_hbm_grid.get_tile_at_index((h_i,))
            actual_h = wgt_hbm_tile.shape[-1]
            dma.Load(
                tile_stream.tile(wght_bufs[h_i], (d, actual_h), tile_dims=wgt_tile_dims, iter_order=ColMajor()),
                tile_stream.tile(wgt_hbm_tile, (n * d, actual_h)),
            ).execute()

    scale_hbm_grid = None
    if is_row:
        scale_hbm_view = weight_scale.slice(1, h_start, h_start + h_sharded)
        scale_hbm_grid = tile_stream.tile(scale_hbm_view, (P_MAX, h_block_size))

    # Load bias and broadcast
    bias_buf = None
    if bias is not None:
        # Bias is per-H (1, h_sharded), broadcast to bxs_pmax rows (reused for each output P-tile)
        bias_loaded = tile_stream.alloc_logical((1, h_sharded), 1, bias.dtype, "bias_loaded", sbm=sbm)
        dma.load(bias_loaded, bias.slice(1, h_start, h_start + h_sharded))
        bias_buf = tile_stream.alloc_logical((bxs_pmax, h_sharded), bxs_pmax, bias.dtype, "bs", sbm=sbm)
        blas.broadcast(bias_buf, bias_loaded)

    out_hbm_grid = tile_stream.tile(output.slice(1, h_start, h_start + h_sharded), (bxs_pmax, h_sharded))

    # Open scope for output multi-buffering
    sbm.open_scope(interleave_degree=out_sb_interleave_degree, name="output_tile_loop")
    for bxs_tile in TiledRange(bxs, P_MAX):
        out_buf = tile_stream.alloc_logical(
            (bxs_tile.size, h_sharded), bxs_tile.size, attention.dtype, f"out_{bxs_tile.index}", sbm=sbm
        )

        attn_outer_tile = (n // 2 * d, 2, bxs_pmax) if use_double_row else (n * d, bxs_pmax)
        attn_pos = (0, 0, bxs_tile.index) if use_double_row else (0, bxs_tile.index)
        attn_tile = tile_stream.tile(attn_buf, attn_outer_tile).ltile_at(attn_pos)

        for h_i in affine_range(num_h_blocks):
            w_slot = h_i % num_w_h_blocks

            # On-demand weight loading
            if not all_weights_preloaded:
                wgt_hbm_tile = wgt_hbm_grid.get_tile_at_index((h_i,))
                actual_h = wgt_hbm_tile.shape[-1]
                dma.Load(
                    tile_stream.tile(wght_bufs[w_slot], (d, actual_h), tile_dims=wgt_tile_dims, iter_order=ColMajor()),
                    tile_stream.tile(wgt_hbm_tile, (n * d, actual_h)),
                ).execute()

            # ROW dequant scale
            if is_row:
                scale_tile = scale_hbm_grid.get_tile_at_index((h_i,))
                actual_h_scale = scale_tile.shape[-1]
                dequant_scale_buf = tile_stream.alloc_logical(
                    (P_MAX, actual_h_scale), P_MAX, nl.float32, f"dequant_scale_{bxs_tile.index}_{h_i}", sbm=sbm
                )
                dma.load(dequant_scale_buf, scale_tile)

            moving_tile = (d_packed, 2, F_MAX) if use_double_row else (d_packed, F_MAX)
            stationary_tile = (d_packed, 2, bxs_tile.size) if use_double_row else (d_packed, bxs_tile.size)
            dequant_tile = (bxs_tile.size, 1) if is_static else (bxs_tile.size, F_MAX)

            out_tile = tile_stream.tile(out_buf, (bxs_tile.size, h_block_size)).ltile_at((0, h_i))
            bias_tile = tile_stream.tile(bias_buf, (bxs_pmax, h_block_size)).ltile_at((0, h_i)) if bias_buf else None

            blas.Matmul(
                dst=tile_stream.tile(out_tile, (bxs_tile.size, F_MAX), iter_order=RowMajor()),
                moving=tile_stream.tile(wght_bufs[w_slot], moving_tile, iter_order=ColMajor()),
                stationary=tile_stream.tile(attn_tile, stationary_tile, iter_order=ColMajor()),
                bias=tile_stream.tile(bias_tile, (bxs_tile.size, F_MAX), iter_order=RowMajor()) if bias_tile else None,
                dequant_scale=tile_stream.tile(dequant_scale_buf, dequant_tile, iter_order=RowMajor()),
                dequant_type=quantization_type,
                perf_mode="double_row" if use_double_row else None,
                psum_buffer_degree=None if sbm.is_auto_alloc() else NUM_PSUM_BANKS,
            ).execute()

        # Store using HBM grid
        dma.store(dst=out_hbm_grid.tile_at((bxs_tile.index,)), src=out_buf)
        sbm.increment_section()

    sbm.close_scope()

    return output


def _output_projection_transposed(attention, weight, bias, quantization_type, weight_scale, input_scale, sbm):
    """Output: [H0, LNC, H1, B*S] where H0=128, H1=H_sharded/128"""
    d, b, n, s = attention.shape
    _, h = weight.shape
    bxs = b * s

    _, lnc, lnc_id = get_program_sharding_info()
    h_sharded = h // lnc
    h_start = lnc_id * h_sharded
    h0, h1 = P_MAX, h_sharded // P_MAX

    # Head packing: pack multiple heads into partition dimension
    if d % 32 == 0:
        n_packed, d_packed, group_size = calculate_head_packing(n, d, P_MAX)
    else:
        n_packed, d_packed, group_size = n, d, 1

    # Tiling strategy depends on bxs vs F_MAX
    if bxs <= F_MAX:
        # Pack multiple h1 values into one PSUM
        packed_f_tile = max_tile(h1 * bxs, bxs, F_MAX)
        bxs_tile = bxs
    else:
        # Tile bxs into F_MAX chunks, one h1 per PSUM
        packed_f_tile = F_MAX
        bxs_tile = F_MAX
    num_packed = packed_f_tile // bxs_tile
    is_static = quantization_type == QuantizationType.STATIC
    is_row = quantization_type == QuantizationType.ROW

    # Double row optimization: STATIC only, even packed heads, N_packed*B divisible by 32, B*S >= 64
    use_double_row = is_static and n_packed % 2 == 0 and n_packed * b % 32 == 0 and bxs >= 64

    output = nl.ndarray((h0, lnc, h1, bxs), dtype=attention.dtype, buffer=nl.shared_hbm)

    # Load scales and compute combined_scale before shuffle
    input_scale_buf, dequant_scale_buf = (
        _load_static_scales(weight_scale, input_scale, sbm) if is_static else (None, None)
    )

    # Load, quantize (if STATIC), and shuffle attention
    attn_buf = _load_and_shuffle_attn(attention, input_scale_buf, quantization_type, d_packed, use_double_row, sbm)

    if use_double_row:
        # Double row: load each head separately into (d, 2, h0, h1) structure
        wght_view = weight.slice(1, h_start, h_start + h_sharded).reshape_dim(1, (h0, h1))
        wght_buf = tile_stream.alloc_logical((n * d // 2, 2, h0, h1), d_packed, weight.dtype, "wght", sbm=sbm)
        dma.Load(
            tile_stream.tile(wght_buf, (d, h0, h1), tile_dims=(0, 2, 3), iter_order=ColMajor()),
            wght_view,
        ).execute()
    else:
        # Weight is (n*d, H), need to load into SBUF container (d_packed, n_p_tiles, h0, h1)
        # Reshape HBM to match: (n*d, h0, h1) -> (n_p_tiles, d_packed, h0, h1) -> permute -> (d_packed, n_p_tiles, h0, h1)
        n_p_tiles = div_ceil(n * d, d_packed)
        wght_view = (
            weight.slice(1, h_start, h_start + h_sharded)
            .reshape_dim(1, (h0, h1))
            .reshape_dim(0, (n_p_tiles, d_packed))
            .permute((1, 0, 2, 3))
        )
        wght_buf = tile_stream.alloc_logical((n * d, h0, h1), d_packed, weight.dtype, "wght", sbm=sbm)
        dma.load(wght_buf, wght_view)

    if is_row:
        dequant_scale_buf = tile_stream.alloc_logical((h0, h1), h0, nl.float32, "dequant_scale", sbm=sbm)
        dma.load(
            dequant_scale_buf,
            weight_scale.select(0, 0).slice(0, h_start, h_start + h_sharded).reshape_dim(0, (h0, h1)),
        )

    bias_buf = None
    if bias is not None:
        bias_sliced = bias.slice(1, h_start, h_start + h_sharded)
        bias_reshaped = bias_sliced.reshape_dim(1, (h0, h1))
        bias_view = bias_reshaped.squeeze_dim(0)
        bias_buf = tile_stream.alloc_logical((h0, h1), h0, bias.dtype, "bs", sbm=sbm)
        dma.load(bias_buf, bias_view)

    out_buf = tile_stream.alloc_logical((h0, h1 * bxs), h0, attention.dtype, "out", sbm=sbm)

    # Configure tiling based on double_row
    moving_tile = (d_packed, 2, bxs_tile) if use_double_row else (d_packed, bxs_tile)
    stationary_tile, stationary_dims = ((d_packed, 2, h0), (0, 1, 2)) if use_double_row else ((d_packed, h0), (0, 1))
    dequant_tile = (h0, 1) if use_double_row or is_static else (h0, num_packed)
    dequant_view = None if use_double_row or is_static else view().broadcast(-1, bxs_tile)

    # Bias: bxs > F_MAX needs virtual_grid to repeat each h1 tile for all bxs chunks
    if bxs > F_MAX:
        bias_ts = tile_stream.tile(
            bias_buf,
            (h0, 1),
            virtual_grid=(div_ceil(bxs, bxs_tile),),
            tile_view=view().broadcast(-1, packed_f_tile),
            iter_order=ColMajor(),
        )
    else:
        bias_ts = tile_stream.tile(
            bias_buf, (h0, num_packed), tile_view=view().broadcast(-1, bxs_tile), iter_order=RowMajor()
        )

    blas.Matmul(
        dst=tile_stream.tile(out_buf, (h0, packed_f_tile), iter_order=RowMajor()),
        moving=tile_stream.tile(attn_buf, moving_tile, iter_order=ColMajor()),
        stationary=tile_stream.tile(wght_buf, stationary_tile, tile_dims=stationary_dims, iter_order=ColMajor()),
        bias=bias_ts,
        dequant_scale=tile_stream.tile(dequant_scale_buf, dequant_tile, tile_view=dequant_view, iter_order=RowMajor()),
        dequant_type=quantization_type,
        perf_mode="double_row" if use_double_row else None,
        psum_buffer_degree=None if sbm.is_auto_alloc() else NUM_PSUM_BANKS,
    ).execute()

    dma.store(dst=output.select(1, lnc_id), src=out_buf.reshape_dim(1, (h1, bxs)))

    return output
