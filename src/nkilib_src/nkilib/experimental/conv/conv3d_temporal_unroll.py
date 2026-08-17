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

"""Temporal-unroll optimization for conv3d with small C_out.

When D_out * C_out is small (e.g., 9), this kernel processes all D_out temporal
positions simultaneously by stacking their filters along the free dimension.
This increases PE utilization from C_out/128 to (D_out*C_out)/128, and reduces
DMA by loading each input depth slice only once instead of once per d_out.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type, get_verified_program_sharding_info

_COL_TILE = 32  # Column tile size for nc_matmul tile_size=(P_MAX, 32)
_MIN_C_IN_FOR_TEMPORAL_UNROLL = 64  # Below this, filter reuse doesn't amortize DMA overhead
_MIN_D_OUT_FOR_TEMPORAL_UNROLL = 2  # Need at least 2 d_out positions for temporal unroll benefit


def should_use_temporal_unroll(C_out: int, D_out: int, C_in: int, K_d: int, W_out: int) -> bool:
    """
    Advisory check: whether conv3d_temporal_unroll is applicable for this shape.

    Callers should invoke this before calling conv3d_temporal_unroll directly.
    Returns True when the problem shape benefits from temporal unrolling:
    D_out temporal positions fit in a single PSUM bank column-tiled,
    C_in is large enough to amortize filter caching, and W_out exceeds F_MAX
    so multiple W tiles are needed (where baseline is slow).

    Args:
        C_out (int): Number of output channels.
        D_out (int): Number of output depth positions.
        C_in (int): Number of input channels.
        K_d (int): Filter depth dimension.
        W_out (int): Output width.

    Returns:
        bool: True if temporal unroll should be used.
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    return (
        D_out * _COL_TILE <= P_MAX
        and C_out <= _COL_TILE
        and D_out >= _MIN_D_OUT_FOR_TEMPORAL_UNROLL
        and C_in > _MIN_C_IN_FOR_TEMPORAL_UNROLL
        and W_out > F_MAX
    )


@nki.jit
def conv3d_temporal_unroll(
    x_in: nl.ndarray,
    filters: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> nl.ndarray:
    """
    3D convolution with temporal unrolling and column tiling for small C_out.

    Processes all D_out temporal output positions simultaneously per (h_out, w_tile),
    improving PE utilization for configurations where C_out is small (e.g., 3). Uses
    column tiling (tile_size=(P_MAX, 32)) to place each d_out in a separate column tile
    for concurrent execution on the systolic array.

    Invoked directly (not dispatched from conv3d entry point). Use
    `should_use_temporal_unroll()` to check applicability before calling.

    Intended Usage Range (enforced by trace-time kernel_assert):
        C_out: 1-32, D_out: 2-4 (D_out * _COL_TILE <= P_MAX, i.e. D_out * 32 <= 128)
        C_in: 65-128 (single C_in tile, K_REP=1)
        W_out: 513+ (multi-W-tile case where baseline underperforms)

    Dimensions:
        B: Batch size
        C_in: Number of input channels
        C_out: Number of output channels
        D: Input depth
        H: Input height
        W: Input width
        K_d: Filter depth
        K_h: Filter height
        K_w: Filter width
        D_out: Output depth = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
        H_out: Output height = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
        W_out: Output width = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1

    Args:
        x_in (nl.ndarray): [B, C_in, D, H, W], Input tensor on HBM.
        filters (nl.ndarray): [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM.
        bias (Optional[nl.ndarray]): [C_out], Optional bias tensor on HBM.
        stride (tuple[int, int, int]): (stride_d, stride_h, stride_w), Convolution strides.
        padding (tuple[int, int, int, int, int, int]): (pad_d_left, pad_d_right, pad_h_top,
            pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
        dilation (tuple[int, int, int]): (dilation_d, dilation_h, dilation_w), Dilation factors.
        activation_fn (Optional[ActFnType]): Optional activation function to apply after conv.
        lnc_shard (bool): Enable LNC sharding across neuron cores (shards on H_out).

    Returns:
        y_out (nl.ndarray): [B, C_out, D_out, H_out, W_out], Output tensor on HBM.

    Pseudocode:
        precompute din_to_pairs[d_in] = [(d_out, k_d), ...] at trace time
        cache all filters[k_d, k_h, k_w] in SBUF once per batch

        for h_out, w_tile:
            allocate PSUM[P_MAX, w_tile_size] with column tiling (_COL_TILE=32)
            for c_in_tile:
                for d_in:
                    load input_block[C_in, h_window * w_window] from HBM
                    for k_h, k_w:
                        input_slice = input_block[:, row_offset + kw_offset : ... : stride_w]
                        for (d_out, k_d) in din_to_pairs[d_in]:
                            nc_matmul(
                                dst=PSUM[nl.ds(d_out*32, C_out), :],
                                stationary=filter[k_d][k_h][k_w],
                                moving=input_slice,
                                tile_position=(0, d_out*32),
                            )
            for d_out:
                tensor_copy PSUM[d_out*32 : d_out*32+C_out] → SBUF
                apply bias + activation
                DMA store to y_out[:, :, d_out, h_out, w_start:w_end]
    """
    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = x_in.shape
    K_d, K_h, K_w, _, C_out = filters.shape
    D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1

    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax

    kernel_assert(
        C_out <= _COL_TILE,
        f"conv3d_temporal_unroll requires C_out <= {_COL_TILE}, got C_out={C_out}",
    )
    kernel_assert(
        D_out * _COL_TILE <= P_MAX,
        f"conv3d_temporal_unroll requires D_out * {_COL_TILE} <= {P_MAX}, got D_out={D_out} (D_out*{_COL_TILE}={D_out * _COL_TILE})",
    )
    W_TILE_SIZE = min(W_out, F_MAX)

    C_IN_TILE_COUNT = div_ceil(C_in, P_MAX)

    # Spatial window sizes
    w_window = (W_TILE_SIZE - 1) * stride_w + (K_w - 1) * dilation_w + 1
    h_window = (K_h - 1) * dilation_h + 1

    y_out = nl.ndarray(
        shape=(B, C_out, D_out, H_out, W_out),
        dtype=x_in.dtype,
        buffer=nl.shared_hbm,
    )

    # LNC sharding on H_out
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d_tunroll", (0, 1))
    h_per_nc = div_ceil(H_out, n_prgs)
    h_start = h_per_nc * prg_id
    h_end = min(h_start + h_per_nc, H_out)

    # --- PRECOMPUTE: d_in → valid (d_out, k_d) pairs ---
    din_to_pairs = []
    for d_in in range(D):
        pairs = []
        for d_out in range(D_out):
            k_d_num = d_in * dilation_d + pad_d_left - d_out * stride_d
            if k_d_num < 0 or k_d_num >= K_d * dilation_d:
                continue
            if k_d_num % dilation_d != 0:
                continue
            k_d = k_d_num // dilation_d
            pairs.append((d_out, k_d))
        din_to_pairs.append(pairs)

    for batch_idx in range(B):
        # --- Cache all filters in SBUF ONCE per batch (they don't change) ---
        # Indexed as filter_cache[c_in_tile_idx][k_d][k_h][k_w]
        all_filter_cache = []
        for c_in_tile_idx in range(C_IN_TILE_COUNT):
            c_in_start = c_in_tile_idx * P_MAX
            c_in_end = min(c_in_start + P_MAX, C_in)
            c_in_size = c_in_end - c_in_start
            tile_filters = []
            for k_d_idx in range(K_d):
                k_d_filters = []
                for k_h_idx in range(K_h):
                    k_h_filters = []
                    for k_w_idx in range(K_w):
                        filter_buf = nl.ndarray(
                            shape=(c_in_size, C_out),
                            dtype=x_in.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.dma_copy(
                            dst=filter_buf,
                            src=filters[k_d_idx, k_h_idx, k_w_idx, c_in_start:c_in_end, 0:C_out],
                        )
                        k_h_filters.append(filter_buf)
                    k_d_filters.append(k_h_filters)
                tile_filters.append(k_d_filters)
            all_filter_cache.append(tile_filters)

        for h_out in nl.affine_range(h_start, h_end):
            for w_start in nl.affine_range(0, W_out, W_TILE_SIZE):
                w_end = min(w_start + W_TILE_SIZE, W_out)
                w_tile_size = w_end - w_start

                # Column tiling: single PSUM bank, each d_out at a column-tile-aligned offset
                psum_bank = nl.ndarray(
                    shape=(P_MAX, w_tile_size),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.memset(dst=psum_bank, value=0.0)

                # W field bounds (constant for this w_tile)
                field_w_start = w_start * stride_w - pad_w_left
                field_w_end = field_w_start + w_window
                valid_w_start_global = max(0, field_w_start)
                valid_w_end_global = min(W, field_w_end)
                w_load_offset = valid_w_start_global - field_w_start
                w_load_size = valid_w_end_global - valid_w_start_global
                needs_w_pad = w_load_offset > 0 or w_load_offset + w_load_size < w_window

                for c_in_tile_idx in nl.sequential_range(C_IN_TILE_COUNT):
                    c_in_start = c_in_tile_idx * P_MAX
                    c_in_end = min(c_in_start + P_MAX, C_in)
                    c_in_size = c_in_end - c_in_start
                    filter_cache = all_filter_cache[c_in_tile_idx]

                    # --- Main compute loop: one DMA per d_in covering all K_h rows ---
                    for d_in in range(D):
                        pairs = din_to_pairs[d_in]
                        if not pairs:
                            continue

                        # Compute H field for this d_in
                        h_field_start = h_out * stride_h - pad_h_top
                        h_field_end = h_field_start + h_window
                        valid_h_start = max(0, h_field_start)
                        valid_h_end = min(H, h_field_end)

                        if valid_h_start >= valid_h_end:
                            continue

                        h_load_offset = valid_h_start - h_field_start
                        h_load_size = valid_h_end - valid_h_start

                        # Load 2D input block: [C_in, h_window, w_window]
                        input_block = nl.ndarray(
                            shape=(c_in_size, h_window * w_window),
                            dtype=x_in.dtype,
                            buffer=nl.sbuf,
                        )

                        # Zero the block if padding is needed on any edge
                        needs_h_pad = h_load_offset > 0 or h_load_offset + h_load_size < h_window
                        if needs_h_pad or needs_w_pad:
                            nisa.memset(dst=input_block, value=0.0)

                        # Load valid rows
                        for h_local_idx in range(h_load_size):
                            h_in = valid_h_start + h_local_idx
                            row_offset = (h_load_offset + h_local_idx) * w_window + w_load_offset
                            input_block[:, row_offset : row_offset + w_load_size] = nl.load(
                                x_in[
                                    batch_idx, c_in_start:c_in_end, d_in, h_in, valid_w_start_global:valid_w_end_global
                                ]
                            )

                        # Iterate K_h, K_w using slices of loaded block
                        for k_h_idx in range(K_h):
                            row_start = k_h_idx * dilation_h * w_window

                            for k_w_idx in range(K_w):
                                kw_offset = k_w_idx * dilation_w
                                slice_start = row_start + kw_offset
                                input_slice = input_block[
                                    :, slice_start : slice_start + w_tile_size * stride_w : stride_w
                                ]

                                for pair_idx in range(len(pairs)):
                                    d_out = pairs[pair_idx][0]
                                    k_d_val = pairs[pair_idx][1]
                                    nisa.nc_matmul(
                                        dst=psum_bank[nl.ds(d_out * _COL_TILE, C_out), :w_tile_size],
                                        stationary=filter_cache[k_d_val][k_h_idx][k_w_idx],
                                        moving=input_slice,
                                        tile_size=(P_MAX, _COL_TILE),
                                        tile_position=(0, d_out * _COL_TILE),
                                    )

                # Store results: extract each d_out column tile from PSUM and write to HBM
                for d_out_idx in range(D_out):
                    psum_tile = psum_bank[nl.ds(d_out_idx * _COL_TILE, C_out), :w_tile_size]
                    result_sbuf = nl.ndarray(shape=(C_out, w_tile_size), dtype=x_in.dtype, buffer=nl.sbuf)

                    if bias != None and activation_fn != None:
                        bias_sbuf = nl.ndarray(shape=(C_out, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=bias_sbuf, src=bias[0:C_out].reshape((C_out, 1)))
                        nisa.tensor_scalar(dst=result_sbuf, data=psum_tile, op0=nl.add, operand0=bias_sbuf)
                        nisa.activation(dst=result_sbuf, data=result_sbuf, op=get_nl_act_fn_from_type(activation_fn))
                    elif bias != None:
                        bias_sbuf = nl.ndarray(shape=(C_out, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=bias_sbuf, src=bias[0:C_out].reshape((C_out, 1)))
                        nisa.tensor_scalar(dst=result_sbuf, data=psum_tile, op0=nl.add, operand0=bias_sbuf)
                    elif activation_fn != None:
                        nisa.activation(dst=result_sbuf, data=psum_tile, op=get_nl_act_fn_from_type(activation_fn))
                    else:
                        nisa.tensor_copy(dst=result_sbuf, src=psum_tile)

                    nisa.dma_copy(
                        dst=y_out[batch_idx, 0:C_out, d_out_idx, h_out, w_start:w_end],
                        src=result_sbuf,
                    )

    return y_out
