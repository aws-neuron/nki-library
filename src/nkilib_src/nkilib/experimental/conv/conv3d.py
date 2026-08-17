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

"""3D Convolution kernel for NeuronCore with K-replication strategy."""

from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    div_ceil,
    get_nl_act_fn_from_type,
    get_verified_program_sharding_info,
)
from ...core.utils.logging import get_logger

_PARTITION_STRIDE_32 = 32
_PARTITION_STRIDE_64 = 64
_MAX_K_REP_AT_32 = 4
_MAX_K_REP_AT_64 = 2
_PSUM_BANK_SIZE = nl.tile_size.psum_fmax * 4
_NUM_PSUM_BANKS = 8
_BIAS_DTYPE_SIZE = sizeinbytes(nl.float32)
_HEAP_ALIGN = nl.tile_size.sbuf_min_align


@nki.jit
def conv3d(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> nl.NkiTensor:
    """
    3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

    Implements a 3D convolution operation (x_in * filters + bias) optimized for NeuronCore
    using a K-replication strategy for filter loading and W-contiguous tiling for output
    computation. Supports configurable stride, padding, dilation, optional bias, optional
    activation function, and LNC sharding.

    Intended Usage Range:
        B: 1-128
        C_in: 3-1280, C_out: 3-2048
        D: 1-1024, H: 1-1024, W: 1-1024
        K_d: 1-64, K_h: 1-64, K_w: 1-64
        Stride: 1-64 per dimension
        Dilation: 1-64 per dimension

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
        x_in (nl.NkiTensor): [B, C_in, D, H, W], Input tensor on HBM.
        filters (nl.NkiTensor): [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM.
        bias (Optional[nl.NkiTensor]): [C_out], Optional bias tensor on HBM.
        stride (tuple[int, int, int]): (stride_d, stride_h, stride_w), Convolution strides.
        padding (tuple[int, int, int, int, int, int]): (pad_d_left, pad_d_right, pad_h_top,
            pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
        dilation (tuple[int, int, int]): (dilation_d, dilation_h, dilation_w), Dilation factors.
        activation_fn (Optional[ActFnType]): Optional activation function to apply after conv.
        lnc_shard (bool): Enable LNC sharding across neuron cores.

    Returns:
        y_out (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], Output tensor on HBM.

    Pseudocode:
        cfg = build_config(x_in, filters, bias, stride, padding, dilation)
        tile_cfg = build_tile_config(cfg)
        mem_cfg = build_memory_config(cfg, tile_cfg)
        allocate SBUF buffers (filters, bias, results, input windows, stacked inputs)

        for c_out_group in range(0, C_out, c_out_interleave * P_MAX):
            load filters + bias for c_out_group into SBUF

            for batch in range(B):
                for dh_group in range(dh_start, dh_end, num_dh_stacked):
                    for w_tile in range(0, W_out, W_tile):
                        allocate PSUM banks for output tiles

                        for c_in_tile in range(0, C_in, P_MAX):
                            DMA load input window [C_in_tile, D_win, H_win, W_win] from HBM to SBUF
                            for k_outer_tile in range(K_outer_tile_count):
                                scatter input window to stacked layout via tensor_copy
                                nc_matmul: stationary=filters, moving=stacked_input, dst=PSUM

                        apply bias + activation, copy PSUM to result SBUF
                        DMA store result SBUF to HBM

        free SBUF buffers

    Tiling Strategy:
        Loop nest: C_out groups -> Batch -> D-H groups -> W tiles -> (C_in tiles in _conv3d_output_tile)

        SBUF Memory Layout:
        - Filters: [c_out_interleave] x [c_in_tile_count] x [K_outer_tile_count] buffers,
          each (stacked_filter_dim_max, c_out_tile_size_max). Loaded once per C_out group.
        - Bias: [c_out_interleave] buffers, (c_out_tile_size_max, 1), float32.
          Loaded once per C_out group alongside filters.
        - Result SBUF: [w_out_interleave] x [c_out_interleave] buffers,
          each (c_out_tile_size_max, num_dh_stacked * W_tile).
          Rotated across W tiles for pipelining DMA stores with compute.
        - Input windows: [input_window_interleave] buffers,
          each (c_in_tile_size_max, d_window_max, h_window_max, w_window_max).
          Multi-buffered to overlap HBM loads with SBUF scatter.
        - Stacked inputs: [stacked_input_interleave] x [K_outer_tile_count] buffers,
          each (stacked_filter_dim_max, num_dh_stacked * W_tile).
          Multi-buffered to overlap scatter with matmul.

        Engine pipelining: tensor_copy and memset alternate between vector/scalar and
        gpsimd/vector engines respectively to enable concurrent execution.

        Tile Size Selection:
        - W_tile: min(W_out, F_MAX), reduced if memory does not fit
        - num_dh_stacked: packs multiple (d_out, h_out) on free dim when W_out < F_MAX
        - c_out/c_in tile: min(P_MAX, channel_count)
        - K_REP: replicates filter positions within partition stride for small C_in tiles
        - All sizes iteratively reduced in _build_tile_config until SBUF budget is met
    """
    cfg = _build_conv3d_config(x_in, filters, bias, stride, padding, dilation, activation_fn, lnc_shard)
    dtype_size = sizeinbytes(x_in.dtype)
    tile_cfg = _build_tile_config(cfg, lnc_shard, dtype_size)
    mem_cfg = _build_memory_config(cfg, tile_cfg, dtype_size)
    _validate_conv3d_inputs(x_in, filters, bias, cfg)

    y_out = nl.ndarray(
        shape=(cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
        dtype=x_in.dtype,
        buffer=nl.shared_hbm,
    )

    sbm = SbufManager(0, mem_cfg.total_memory_required, logger=get_logger("conv3d"))

    pre_bias_bufs, pre_filter_bufs = _allocate_filter_bias_buffers(sbm, cfg, tile_cfg, mem_cfg, x_in.dtype)

    max_effective_free = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    result_sbuf_slots = []
    for w_slot_idx in range(mem_cfg.w_out_interleave):
        slot = []
        for c_out_slot_idx in range(mem_cfg.c_out_interleave):
            slot.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.c_out_tile_size_max, max_effective_free),
                    dtype=x_in.dtype,
                    name=f"result_{w_slot_idx}_cout{c_out_slot_idx}",
                )
            )
        result_sbuf_slots.append(slot)

    input_window_slots = []
    for input_window_slot_idx in range(mem_cfg.input_window_interleave):
        input_window_slots.append(
            sbm.alloc_heap(
                shape=(
                    tile_cfg.c_in_tile_size_max,
                    tile_cfg.d_window_max,
                    tile_cfg.h_chunk_h_window_max,
                    tile_cfg.w_window_max,
                ),
                dtype=x_in.dtype,
                name=f"input_window_{input_window_slot_idx}",
            )
        )

    stacked_input_slots = []
    for stacked_input_slot_idx in range(mem_cfg.stacked_input_interleave):
        stacked_input_bufs = []
        for k_outer_idx in range(tile_cfg.K_outer_tile_count):
            stacked_input_bufs.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, max_effective_free),
                    dtype=x_in.dtype,
                    name=f"stacked_input_{stacked_input_slot_idx}_k{k_outer_idx}",
                )
            )
        stacked_input_slots.append(stacked_input_bufs)

    for c_out_group_start in range(tile_cfg.C_out_start, tile_cfg.C_out_end, mem_cfg.c_out_interleave * tile_cfg.P_MAX):
        c_out_group_end = min(c_out_group_start + mem_cfg.c_out_interleave * tile_cfg.P_MAX, tile_cfg.C_out_end)
        actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)

        c_out_tile_sizes, bias_cache, filters_cache = _load_bias_and_filters_for_c_out_group(
            filters,
            bias,
            cfg,
            tile_cfg,
            c_out_group_start,
            c_out_group_end,
            pre_bias_bufs,
            pre_filter_bufs,
        )

        w_out_tile_counter = 0
        c_in_slot_counter = 0

        for batch_idx in range(cfg.B):
            x_in_batch = x_in.select(dim=0, index=batch_idx)

            chunk_preloaded_windows = None
            dh_group_idx = tile_cfg.dh_start
            while dh_group_idx < tile_cfg.dh_end:
                dh_positions = []
                first_d_out = None
                for dh_stacked_idx in range(tile_cfg.num_dh_stacked):
                    flat_idx = dh_group_idx + dh_stacked_idx
                    if flat_idx >= tile_cfg.dh_end:
                        break
                    d_out_idx, h_out_idx = divmod(flat_idx, cfg.H_out)
                    if first_d_out == None:
                        first_d_out = d_out_idx
                    elif d_out_idx != first_d_out:
                        break  # don't cross D boundary — avoids window spanning full H
                    dh_positions.append((d_out_idx, h_out_idx))
                num_dh_positions = len(dh_positions)

                # Determine if this dh position participates in H-chunking.
                d_out_current = dh_positions[0][0]
                h_out_current = dh_positions[0][1]
                h_chunk_enabled = tile_cfg.h_chunk_size > 1 and tile_cfg.num_dh_stacked == 1
                is_chunk_start = h_chunk_enabled and (h_out_current % tile_cfg.h_chunk_size == 0)

                # Build the chunk's dh_positions
                if is_chunk_start:
                    chunk_dh_positions = []
                    for h_chunk_idx in range(tile_cfg.h_chunk_size):
                        chunk_h_out = h_out_current + h_chunk_idx
                        if chunk_h_out >= cfg.H_out:
                            break
                        # Check we don't cross d_out boundary
                        chunk_flat = d_out_current * cfg.H_out + chunk_h_out
                        if chunk_flat >= tile_cfg.dh_end:
                            break
                        chunk_dh_positions.append((d_out_current, chunk_h_out))
                    actual_chunk_size = len(chunk_dh_positions)

                for w_start in range(0, cfg.W_out, tile_cfg.W_tile):
                    w_end = min(w_start + tile_cfg.W_tile, cfg.W_out)
                    w_tile_size = w_end - w_start
                    effective_free_dim = num_dh_positions * w_tile_size

                    # If this is the first h_out in an H-chunk, load the chunk's
                    # union input window for all c_in tiles.
                    if is_chunk_start:
                        chunk_preloaded_windows = []
                        c_in_tile_idx_load = 0
                        for c_in_start_load in range(0, cfg.C_in, tile_cfg.P_MAX):
                            c_in_end_load = min(c_in_start_load + tile_cfg.P_MAX, cfg.C_in)
                            global_idx_load = c_in_slot_counter + c_in_tile_idx_load
                            input_window_buf_load = input_window_slots[global_idx_load % len(input_window_slots)]
                            x_in_cin_load = x_in_batch.slice(dim=0, start=c_in_start_load, end=c_in_end_load, step=1)
                            loaded = _load_input_window_to_sbuf_3d(
                                x_in_cin=x_in_cin_load,
                                input_window_buf=input_window_buf_load,
                                cfg=cfg,
                                dh_positions=chunk_dh_positions,
                                w_start=w_start,
                                w_end=w_end,
                            )
                            chunk_preloaded_windows.append(loaded)
                            c_in_tile_idx_load += 1

                    slot_idx = w_out_tile_counter % mem_cfg.w_out_interleave
                    result_sbufs = result_sbuf_slots[slot_idx][:actual_group_size]
                    psum_bank_idx = (slot_idx * actual_group_size) % _NUM_PSUM_BANKS

                    # Pass preloaded windows if we're in an H-chunk
                    preloaded = chunk_preloaded_windows if h_chunk_enabled else None

                    _conv3d_output_tile(
                        x_in_batch=x_in_batch,
                        filters_cache=filters_cache,
                        bias_cache=bias_cache,
                        result_sbufs=result_sbufs,
                        input_window_slots=input_window_slots,
                        stacked_input_slots=stacked_input_slots,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                        psum_bank_idx=psum_bank_idx,
                        c_in_slot_offset=c_in_slot_counter,
                        preloaded_windows=preloaded,
                    )
                    c_in_slot_counter += tile_cfg.c_in_tile_count

                    # Store results from SBUF to HBM
                    for c_out_tile_idx in range(actual_group_size):
                        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
                        c_out_tile_end = c_out_tile_start + c_out_tile_sizes[c_out_tile_idx]

                        result_view = (
                            (result_sbufs[c_out_tile_idx])
                            .slice(dim=0, start=0, end=c_out_tile_sizes[c_out_tile_idx], step=1)
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_positions, w_tile_size))
                        )
                        total_dh = cfg.D_out * cfg.H_out
                        y_out_view = (
                            y_out.select(dim=0, index=batch_idx)
                            .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                            .flatten_dims(1, 3)
                            .reshape_dim(1, (total_dh, cfg.W_out))
                            .slice(dim=1, start=dh_group_idx, end=dh_group_idx + num_dh_positions, step=1)
                            .slice(dim=2, start=w_start, end=w_end, step=1)
                        )
                        nisa.dma_copy(
                            dst=y_out_view,
                            src=result_view,
                        )

                    w_out_tile_counter += 1
                dh_group_idx += num_dh_positions

    # Free all heap allocations
    for _ in range(mem_cfg.stacked_input_interleave):
        for _ in range(tile_cfg.K_outer_tile_count):
            sbm.pop_heap()
    for _ in range(mem_cfg.input_window_interleave):
        sbm.pop_heap()
    for _ in range(mem_cfg.w_out_interleave):
        for _ in range(mem_cfg.c_out_interleave):
            sbm.pop_heap()
    for _ in range(tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count):
        sbm.pop_heap()
    for _ in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            sbm.pop_heap()

    return y_out


def _heap_align_waste(bytes_per_partition: int) -> int:
    """Compute alignment waste for a single SBM heap allocation."""
    return (_HEAP_ALIGN - (bytes_per_partition % _HEAP_ALIGN)) % _HEAP_ALIGN


def _decompose_k_position(k_position: int, K_h: int, K_w: int) -> tuple[int, int, int]:
    """Decompose a flat filter position index into (k_d, k_h, k_w) indices."""
    k_d_idx, k_hw_remainder = divmod(k_position, K_h * K_w)
    k_h_idx, k_w_idx = divmod(k_hw_remainder, K_w)
    return k_d_idx, k_h_idx, k_w_idx


def _compute_used_k_positions(cfg: "Conv3dConfig") -> list[int]:
    """
    Compute which flat filter positions are actually used.

    For each filter position (k_d, k_h, k_w), checks whether there exists any
    output position where the corresponding input position falls within valid
    (non-padded) bounds.

    Args:
        cfg: Conv3dConfig with all convolution parameters.

    Returns:
        List of flat filter position indices (into K_d*K_h*K_w) that are actually used.
    """
    K_d, K_h, K_w = cfg.K_d, cfg.K_h, cfg.K_w

    used_k_d = [False] * K_d
    for k_d in range(K_d):
        for d_out in range(cfg.D_out):
            d_in = d_out * cfg.stride_d + k_d * cfg.dilation_d - cfg.pad_d_left
            if 0 <= d_in < cfg.D:
                used_k_d[k_d] = True
                break

    used_k_h = [False] * K_h
    for k_h in range(K_h):
        for h_out in range(cfg.H_out):
            h_in = h_out * cfg.stride_h + k_h * cfg.dilation_h - cfg.pad_h_top
            if 0 <= h_in < cfg.H:
                used_k_h[k_h] = True
                break

    used_k_w = [False] * K_w
    for k_w in range(K_w):
        for w_out in range(cfg.W_out):
            w_in = w_out * cfg.stride_w + k_w * cfg.dilation_w - cfg.pad_w_left
            if 0 <= w_in < cfg.W:
                used_k_w[k_w] = True
                break

    used_positions = []
    for k_d in range(K_d):
        for k_h in range(K_h):
            for k_w in range(K_w):
                if used_k_d[k_d] and used_k_h[k_h] and used_k_w[k_w]:
                    flat_idx = k_d * K_h * K_w + k_h * K_w + k_w
                    used_positions.append(flat_idx)

    return used_positions


@dataclass
class Conv3dConfig(nl.NKIObject):
    """Configuration for 3D convolution kernel parameters and input/output dimensions."""

    stride_d: int
    stride_h: int
    stride_w: int
    pad_d_left: int
    pad_d_right: int
    pad_h_top: int
    pad_h_bottom: int
    pad_w_left: int
    pad_w_right: int
    dilation_d: int
    dilation_h: int
    dilation_w: int
    has_bias: bool
    has_activation: bool
    activation_fn: Optional[ActFnType]
    lnc_shard: bool
    B: int
    C_in: int
    C_out: int
    D: int
    H: int
    W: int
    K_d: int
    K_h: int
    K_w: int
    D_out: int
    H_out: int
    W_out: int


@dataclass
class Conv3dTileConfig(nl.NKIObject):
    """Tiling configuration including partition sizes, tile counts, and sharding ranges."""

    P_MAX: int
    F_MAX: int
    PSUM_BANK_SIZE: int
    NUM_PSUM_BANKS: int
    W_tile: int
    c_out_tile_size_max: int
    c_in_tile_size_max: int
    c_in_tile_count: int
    c_out_tile_count: int
    K_outer_tile_count: int
    K_REP_max: int
    partition_stride_max: int
    stacked_filter_dim_max: int
    window_size_w_max: int
    C_out_start: int
    C_out_end: int
    C_out_local: int
    num_dh_stacked: int
    d_window_max: int
    h_window_max: int
    w_window_max: int
    shard_on_dh: bool
    dh_start: int
    dh_end: int
    used_k_positions: list[int]
    K_total_used: int
    h_chunk_size: int
    h_chunk_h_window_max: int
    # TODO: Determine optimal engine offload values using a cost model.
    tensor_copy_activation_engine_offload: int
    memset_gpsimd_engine_offload: int


@dataclass
class Conv3dMemoryConfig(nl.NKIObject):
    """Memory layout configuration for SBUF buffer allocation and interleaving."""

    dtype_size: int
    TOTAL_SBUF: int
    bias_size_per_tile: int
    filters_per_c_out_tile: int
    per_c_out_tile_outer: int
    per_c_out_tile_inner: int
    input_window_memory: int
    stacked_input_memory: int
    window_size_w_max: int
    c_out_interleave: int
    input_window_interleave: int
    stacked_input_interleave: int
    w_out_interleave: int
    num_w_out_tiles: int
    total_memory_required: int


def _build_conv3d_config(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
    dilation: tuple[int, int, int],
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
) -> Conv3dConfig:
    """
    Build Conv3dConfig from input tensors and convolution parameters.

    Extracts dimensions from input/filter shapes and computes output spatial
    dimensions based on stride, padding, and dilation settings.
    """
    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = x_in.shape
    K_d, K_h, K_w, _, C_out = filters.shape
    D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1
    return Conv3dConfig(
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_d_left=pad_d_left,
        pad_d_right=pad_d_right,
        pad_h_top=pad_h_top,
        pad_h_bottom=pad_h_bottom,
        pad_w_left=pad_w_left,
        pad_w_right=pad_w_right,
        dilation_d=dilation_d,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        has_bias=bias != None,
        has_activation=activation_fn != None,
        activation_fn=activation_fn,
        lnc_shard=lnc_shard,
        B=B,
        C_in=C_in,
        C_out=C_out,
        D=D,
        H=H,
        W=W,
        K_d=K_d,
        K_h=K_h,
        K_w=K_w,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
    )


def _get_k_replication_params(c_in_tile_size: int, K_total: int) -> tuple[int, int]:
    """Determine K-replication count and partition stride based on input channel tile size."""
    if c_in_tile_size <= _PARTITION_STRIDE_32:
        return min(K_total, _MAX_K_REP_AT_32), _PARTITION_STRIDE_32
    elif c_in_tile_size <= _PARTITION_STRIDE_64:
        return min(K_total, _MAX_K_REP_AT_64), _PARTITION_STRIDE_64
    else:
        return 1, c_in_tile_size


def _calculate_union_window_size(cfg: Conv3dConfig, num_dh_stacked: int, w_tile: int) -> tuple[int, int, int]:
    """Calculate the union input window size for a group of stacked D-H output positions."""
    single_d_window = (cfg.K_d - 1) * cfg.dilation_d + 1
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    w_window = (w_tile - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w + 1
    if num_dh_stacked == 1:
        return single_d_window, single_h_window, w_window
    max_d_out_in_group = (num_dh_stacked - 1 + cfg.H_out - 1) // cfg.H_out
    d_window_max = min(single_d_window + max_d_out_in_group * cfg.stride_d, cfg.D)
    max_h_out_in_group = min(num_dh_stacked - 1, cfg.H_out - 1)
    h_window_max = min(single_h_window + max_h_out_in_group * cfg.stride_h, cfg.H)
    return d_window_max, h_window_max, w_window


def _build_tile_config(cfg: Conv3dConfig, lnc_shard: bool, dtype_size: int) -> Conv3dTileConfig:
    """Build tiling configuration with LNC sharding and iterative memory fitting."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    TOTAL_SBUF = nl.tile_size.total_available_sbuf_size
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))

    full_c_out_tile_count = div_ceil(cfg.C_out, P_MAX)
    total_dh_positions = cfg.D_out * cfg.H_out

    dh_work_per_nc = div_ceil(total_dh_positions, 2)
    cout_tiles_per_nc = div_ceil(full_c_out_tile_count, 2)
    dh_waste = 2 * dh_work_per_nc - total_dh_positions
    cout_waste = 2 * cout_tiles_per_nc - full_c_out_tile_count
    shard_on_dh = (total_dh_positions >= 2) and (dh_waste <= cout_waste)

    if lnc_shard and shard_on_dh:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_per_nc = div_ceil(total_dh_positions, n_prgs)
        dh_start = dh_per_nc * prg_id
        dh_end = min(dh_start + dh_per_nc, total_dh_positions)
    elif lnc_shard:
        C_out_per_nc = div_ceil(cfg.C_out, n_prgs)
        C_out_start = C_out_per_nc * prg_id
        C_out_end = min(C_out_start + C_out_per_nc, cfg.C_out)
        C_out_local = C_out_end - C_out_start
        dh_start, dh_end = 0, total_dh_positions
    else:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_start, dh_end = 0, total_dh_positions
        shard_on_dh = False

    c_out_tile_size_max = min(P_MAX, C_out_local)
    c_in_tile_size_max = min(P_MAX, cfg.C_in)
    used_k_positions = _compute_used_k_positions(cfg)
    K_total_used = len(used_k_positions)
    K_REP_max, partition_stride_max = _get_k_replication_params(c_in_tile_size_max, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP_max)
    stacked_filter_dim_max = partition_stride_max * K_REP_max
    c_in_tile_count = div_ceil(cfg.C_in, P_MAX)
    c_out_tile_count = div_ceil(C_out_local, P_MAX)

    W_tile = min(cfg.W_out, F_MAX)
    num_dh_stacked = min(F_MAX // cfg.W_out, total_dh_positions) if cfg.W_out < F_MAX and total_dh_positions > 1 else 1

    while True:
        effective_free_dim = num_dh_stacked * W_tile
        d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)
        bias_mem = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
        filter_mem = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size
        min_mem = (
            bias_mem
            + filter_mem
            + effective_free_dim * dtype_size
            + d_window_max * h_window_max * w_window_max * dtype_size
            + K_outer_tile_count * effective_free_dim * dtype_size
        )
        if min_mem <= TOTAL_SBUF:
            break
        if num_dh_stacked > 1:
            num_dh_stacked = max(1, num_dh_stacked // 2)
        elif W_tile > 1:
            W_tile = max(1, W_tile // 2)
        else:
            break

    window_size_w_max = (W_tile - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w + 1
    d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)

    # Compute H-chunk size for input data reuse across consecutive h_out positions.
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    num_w_tiles = div_ceil(cfg.W_out, W_tile)
    if num_dh_stacked == 1 and cfg.K_h > 1 and cfg.H_out > 1 and num_w_tiles == 1:
        h_chunk_size = cfg.K_h

        h_chunk_h_window = single_h_window + (h_chunk_size - 1) * cfg.stride_h
        h_chunk_h_window = min(h_chunk_h_window, cfg.H)

        effective_free_dim_check = num_dh_stacked * W_tile
        while h_chunk_size > 1:
            h_chunk_h_window = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
            chunk_input_window_mem = d_window_max * h_chunk_h_window * w_window_max * dtype_size
            bias_mem_check = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
            filter_mem_check = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size

            min_mem_check = (
                bias_mem_check
                + filter_mem_check
                + effective_free_dim_check * dtype_size
                + chunk_input_window_mem
                + K_outer_tile_count * effective_free_dim_check * dtype_size
            )
            if min_mem_check <= TOTAL_SBUF:
                break
            h_chunk_size -= 1
        h_chunk_h_window_max = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
    else:
        h_chunk_size = 1
        h_chunk_h_window_max = h_window_max

    tile_cfg = Conv3dTileConfig(
        P_MAX=P_MAX,
        F_MAX=F_MAX,
        PSUM_BANK_SIZE=_PSUM_BANK_SIZE,
        NUM_PSUM_BANKS=_NUM_PSUM_BANKS,
        W_tile=W_tile,
        c_out_tile_size_max=c_out_tile_size_max,
        c_in_tile_size_max=c_in_tile_size_max,
        c_in_tile_count=c_in_tile_count,
        c_out_tile_count=c_out_tile_count,
        K_outer_tile_count=K_outer_tile_count,
        K_REP_max=K_REP_max,
        partition_stride_max=partition_stride_max,
        stacked_filter_dim_max=stacked_filter_dim_max,
        window_size_w_max=window_size_w_max,
        C_out_start=C_out_start,
        C_out_end=C_out_end,
        C_out_local=C_out_local,
        num_dh_stacked=num_dh_stacked,
        d_window_max=d_window_max,
        h_window_max=h_window_max,
        w_window_max=w_window_max,
        shard_on_dh=shard_on_dh,
        dh_start=dh_start,
        dh_end=dh_end,
        used_k_positions=used_k_positions,
        K_total_used=K_total_used,
        h_chunk_size=h_chunk_size,
        h_chunk_h_window_max=h_chunk_h_window_max,
        tensor_copy_activation_engine_offload=4,
        memset_gpsimd_engine_offload=2,
    )

    K_total_full = cfg.K_d * cfg.K_h * cfg.K_w
    logger = get_logger("conv3d")
    logger.debug(
        f"TileConfig: W_tile={tile_cfg.W_tile}, num_dh_stacked={tile_cfg.num_dh_stacked}, "
        f"c_out_tile_size_max={tile_cfg.c_out_tile_size_max}, c_in_tile_size_max={tile_cfg.c_in_tile_size_max}, "
        f"c_in_tile_count={tile_cfg.c_in_tile_count}, c_out_tile_count={tile_cfg.c_out_tile_count}, "
        f"K_outer_tile_count={tile_cfg.K_outer_tile_count}, K_REP_max={tile_cfg.K_REP_max}, "
        f"K_total_used={tile_cfg.K_total_used}/{K_total_full}, "
        f"h_chunk_size={tile_cfg.h_chunk_size}, h_chunk_h_window_max={tile_cfg.h_chunk_h_window_max}, "
        f"shard_on_dh={tile_cfg.shard_on_dh}, dh_range=[{tile_cfg.dh_start},{tile_cfg.dh_end}), "
        f"C_out_range=[{tile_cfg.C_out_start},{tile_cfg.C_out_end})"
    )

    return tile_cfg


def _build_memory_config(cfg: Conv3dConfig, tile_cfg: Conv3dTileConfig, dtype_size: int) -> Conv3dMemoryConfig:
    """Build memory configuration for SBUF buffer allocation and interleaving."""
    TOTAL_SBUF = nl.tile_size.total_available_sbuf_size
    num_w_out_tiles = div_ceil(cfg.W_out, tile_cfg.W_tile)
    bias_size_per_tile = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
    filters_per_c_out_tile = (
        tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * tile_cfg.c_out_tile_size_max * dtype_size
    )
    per_c_out_tile_outer = bias_size_per_tile + filters_per_c_out_tile
    effective_free_dim = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    per_c_out_tile_inner = effective_free_dim * dtype_size
    input_window_memory = tile_cfg.d_window_max * tile_cfg.h_chunk_h_window_max * tile_cfg.w_window_max * dtype_size
    stacked_input_memory = tile_cfg.K_outer_tile_count * effective_free_dim * dtype_size
    bias_align_waste = _heap_align_waste(_BIAS_DTYPE_SIZE) if cfg.has_bias else 0
    result_align_waste = _heap_align_waste(effective_free_dim * dtype_size)
    input_window_align_waste = _heap_align_waste(input_window_memory)
    stacked_input_align_waste = _heap_align_waste(effective_free_dim * dtype_size)
    input_window_cost = input_window_memory + input_window_align_waste
    stacked_input_cost = stacked_input_memory + tile_cfg.K_outer_tile_count * stacked_input_align_waste

    c_out_interleave = 1
    w_out_interleave = _NUM_PSUM_BANKS
    input_window_interleave = 2
    stacked_input_interleave = 2

    found_fit = False
    for min_buf_count in (2, 1):
        if found_fit:
            break
        for try_c_out in range(tile_cfg.c_out_tile_count, 0, -1):
            if found_fit:
                break
            c_out_wide_try = try_c_out * tile_cfg.c_out_tile_size_max
            filter_align_waste_try = _heap_align_waste(c_out_wide_try * dtype_size)
            fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste_try
            outer_cost = try_c_out * per_c_out_tile_outer + try_c_out * bias_align_waste + fixed_filter_align
            for try_w_out in range(_NUM_PSUM_BANKS, 0, -1):
                inner_cost = try_w_out * try_c_out * (per_c_out_tile_inner + result_align_waste)
                min_total = (
                    outer_cost + inner_cost + min_buf_count * input_window_cost + min_buf_count * stacked_input_cost
                )
                if min_total <= TOTAL_SBUF:
                    c_out_interleave, w_out_interleave = try_c_out, try_w_out
                    input_window_interleave = min_buf_count
                    stacked_input_interleave = min_buf_count
                    found_fit = True
                    break

    kernel_assert(
        found_fit,
        f"Failed to find an SBUF-fitting memory configuration for conv3d. "
        f"TOTAL_SBUF={TOTAL_SBUF}, per_c_out_tile_outer={per_c_out_tile_outer}, "
        f"per_c_out_tile_inner={per_c_out_tile_inner}, "
        f"input_window_cost={input_window_cost}, stacked_input_cost={stacked_input_cost}, "
        f"c_out_tile_count={tile_cfg.c_out_tile_count}",
    )

    c_out_wide = c_out_interleave * tile_cfg.c_out_tile_size_max
    filter_align_waste = _heap_align_waste(c_out_wide * dtype_size)
    fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste
    baseline = (
        c_out_interleave * per_c_out_tile_outer
        + c_out_interleave * bias_align_waste
        + fixed_filter_align
        + w_out_interleave * c_out_interleave * (per_c_out_tile_inner + result_align_waste)
        + input_window_interleave * input_window_cost
        + stacked_input_interleave * stacked_input_cost
    )
    remaining = TOTAL_SBUF - baseline

    both_cost = input_window_cost + stacked_input_cost
    while remaining >= both_cost and both_cost > 0:
        input_window_interleave += 1
        stacked_input_interleave += 1
        remaining -= both_cost
    if remaining >= stacked_input_cost and stacked_input_cost > 0:
        stacked_input_interleave += 1
        remaining -= stacked_input_cost
    elif remaining >= input_window_cost and input_window_cost > 0:
        input_window_interleave += 1
        remaining -= input_window_cost

    # Account for SBM heap alignment overhead
    num_bias_allocs = c_out_interleave if cfg.has_bias else 0
    num_filter_allocs = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count
    num_result_allocs = w_out_interleave * c_out_interleave
    num_input_window_allocs = input_window_interleave
    num_stacked_input_allocs = stacked_input_interleave * tile_cfg.K_outer_tile_count

    alignment_overhead = (
        num_filter_allocs * filter_align_waste
        + num_bias_allocs * bias_align_waste
        + num_result_allocs * result_align_waste
        + num_input_window_allocs * input_window_align_waste
        + num_stacked_input_allocs * stacked_input_align_waste
    )

    total_memory_required = (
        w_out_interleave * c_out_interleave * per_c_out_tile_inner
        + input_window_interleave * input_window_memory
        + stacked_input_interleave * stacked_input_memory
        + c_out_interleave * per_c_out_tile_outer
        + alignment_overhead
    )

    mem_cfg = Conv3dMemoryConfig(
        dtype_size=dtype_size,
        TOTAL_SBUF=TOTAL_SBUF,
        bias_size_per_tile=bias_size_per_tile,
        filters_per_c_out_tile=filters_per_c_out_tile,
        per_c_out_tile_outer=per_c_out_tile_outer,
        per_c_out_tile_inner=per_c_out_tile_inner,
        input_window_memory=input_window_memory,
        stacked_input_memory=stacked_input_memory,
        window_size_w_max=tile_cfg.window_size_w_max,
        c_out_interleave=c_out_interleave,
        input_window_interleave=input_window_interleave,
        stacked_input_interleave=stacked_input_interleave,
        w_out_interleave=w_out_interleave,
        num_w_out_tiles=num_w_out_tiles,
        total_memory_required=total_memory_required,
    )

    logger = get_logger("conv3d")
    logger.info(
        f"MemoryConfig: c_out_interleave={mem_cfg.c_out_interleave}, "
        f"w_out_interleave={mem_cfg.w_out_interleave}, "
        f"input_window_interleave={mem_cfg.input_window_interleave}, "
        f"stacked_input_interleave={mem_cfg.stacked_input_interleave}, "
        f"alignment_overhead={alignment_overhead}, "
        f"total_memory_required={mem_cfg.total_memory_required}, TOTAL_SBUF={mem_cfg.TOTAL_SBUF}"
    )

    return mem_cfg


def _get_tensor_copy_engine(idx: int, modulo: int = 4) -> Any:
    """Alternate between vector and scalar engines for tensor copy pipelining.

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    return nisa.scalar_engine if idx % modulo == 0 else nisa.vector_engine


def _get_memset_engine(idx: int, modulo: int = 2) -> Any:
    """Alternate between gpsimd and vector engines for memset pipelining.

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    return nisa.gpsimd_engine if idx % modulo == 0 else nisa.vector_engine


def _validate_conv3d_inputs(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
) -> None:
    """Validate all input parameters for the 3D convolution kernel."""
    kernel_assert(cfg.dilation_d >= 1, f"Dilation D must be >= 1, got {cfg.dilation_d}")
    kernel_assert(cfg.dilation_h >= 1, f"Dilation H must be >= 1, got {cfg.dilation_h}")
    kernel_assert(cfg.dilation_w >= 1, f"Dilation W must be >= 1, got {cfg.dilation_w}")
    kernel_assert(cfg.stride_d >= 1, f"Stride D must be >= 1, got {cfg.stride_d}")
    kernel_assert(cfg.stride_h >= 1, f"Stride H must be >= 1, got {cfg.stride_h}")
    kernel_assert(cfg.stride_w >= 1, f"Stride W must be >= 1, got {cfg.stride_w}")
    kernel_assert(
        cfg.pad_d_left >= 0 and cfg.pad_d_right >= 0,
        f"Padding D non-negative: ({cfg.pad_d_left}, {cfg.pad_d_right})",
    )
    kernel_assert(
        cfg.pad_h_top >= 0 and cfg.pad_h_bottom >= 0,
        f"Padding H non-negative: ({cfg.pad_h_top}, {cfg.pad_h_bottom})",
    )
    kernel_assert(
        cfg.pad_w_left >= 0 and cfg.pad_w_right >= 0,
        f"Padding W non-negative: ({cfg.pad_w_left}, {cfg.pad_w_right})",
    )
    kernel_assert(cfg.D_out >= 1, f"Output depth D_out must be >= 1, got {cfg.D_out}")
    kernel_assert(cfg.H_out >= 1, f"Output height H_out must be >= 1, got {cfg.H_out}")
    kernel_assert(cfg.W_out >= 1, f"Output width W_out must be >= 1, got {cfg.W_out}")
    kernel_assert(x_in.shape[1] == filters.shape[3], f"C_in mismatch: {x_in.shape[1]} vs {filters.shape[3]}")
    kernel_assert(x_in.dtype == filters.dtype, f"dtype mismatch: {x_in.dtype} vs {filters.dtype}")
    if bias != None:
        kernel_assert(x_in.dtype == bias.dtype, f"bias dtype mismatch: {x_in.dtype} vs {bias.dtype}")


# Memory Hierarchy 3: SBUF operations


def _compute_valid_free_range(
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    used_k_positions: list[int],
    k_start: int,
    k_end: int,
) -> tuple[int, int]:
    """
    Compute the contiguous valid free-dim range for a K-outer tile.

    For each K position in [k_start, k_end), determines which (dh_idx, w_out)
    positions in the free dimension have at least one input position within valid
    bounds.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    full_free_dim = num_dh_stacked * w_tile_size

    # Start with empty range and take union
    global_valid_start = full_free_dim
    global_valid_end = 0

    for k_idx in range(k_start, k_end):
        flat_k_pos = used_k_positions[k_idx]
        k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)

        # Compute valid W range for this K position
        base_w_in_first = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        # First valid w_out offset: input position >= 0
        if base_w_in_first < 0:
            k_w_valid_start = div_ceil(-base_w_in_first, cfg.stride_w)
        else:
            k_w_valid_start = 0
        # Last valid w_out offset: input position < W
        base_w_in_last = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        if base_w_in_last >= cfg.W:
            k_w_valid_end = w_tile_size - div_ceil(base_w_in_last - cfg.W + 1, cfg.stride_w)
        else:
            k_w_valid_end = w_tile_size

        if k_w_valid_start >= k_w_valid_end:
            # This K position has no valid W positions
            continue

        # Find first and last valid dh positions for this K position
        first_valid_dh = -1
        last_valid_dh = -1
        for dh_idx in range(num_dh_stacked):
            d_out, h_out = dh_positions[dh_idx]
            d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
            h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
            if 0 <= d_in < cfg.D and 0 <= h_in < cfg.H:
                if first_valid_dh == -1:
                    first_valid_dh = dh_idx
                last_valid_dh = dh_idx

        if first_valid_dh == -1:
            # No valid dh positions for this K position
            continue

        # Valid free-dim range: [first_valid_dh * w_tile_size + k_w_valid_start,
        #                        last_valid_dh * w_tile_size + k_w_valid_end)
        k_valid_start = first_valid_dh * w_tile_size + k_w_valid_start
        k_valid_end = last_valid_dh * w_tile_size + k_w_valid_end

        # Union with global range
        global_valid_start = min(global_valid_start, k_valid_start)
        global_valid_end = max(global_valid_end, k_valid_end)

    if global_valid_start >= global_valid_end:
        return 0, 0

    return global_valid_start, global_valid_end


def _compute_unwritten_free_ranges(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> list[tuple[int, int]]:
    """
    Compute free-dim ranges that no nc_matmul will ever write for this output tile.
    """
    K_total_used = tile_cfg.K_total_used
    K_REP = tile_cfg.K_REP_max
    K_outer_tile_count = tile_cfg.K_outer_tile_count
    w_tile_size = w_end - w_start
    effective_free_dim = len(dh_positions) * w_tile_size

    # Build a written mask over the free dimension by unioning each K-outer tile's
    # valid range. Positions left False are never written by any matmul.
    written = [False] * effective_free_dim
    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        valid_start, valid_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            tile_cfg.used_k_positions,
            k_start,
            k_end,
        )
        for pos in range(valid_start, valid_end):
            written[pos] = True

    # Coalesce contiguous unwritten positions into ranges
    unwritten_ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < effective_free_dim:
        if written[pos]:
            pos += 1
            continue
        gap_start = pos
        while pos < effective_free_dim and not written[pos]:
            pos += 1
        unwritten_ranges.append((gap_start, pos))
    return unwritten_ranges


def _scatter_input_to_stacked(
    input_window: Optional[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    memset_engine_idx: int = 0,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
) -> tuple[list[Optional[nl.NkiTensor]], list[int], list[int]]:
    """
    Scatter input window data into K-replicated stacked buffers for matmul.

    For each K-outer tile, maps the relevant input positions from the loaded
    input window into a stacked buffer layout where multiple filter positions
    are packed along the partition dimension. Handles padding by zero-initializing
    buffers or adjusting the free-dimension and stride of input tiles. Returns None
    for fully-padded K-outer tiles to skip unnecessary matmul operations.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    effective_free_dim = num_dh_stacked * w_tile_size
    K_REP, partition_stride = _get_k_replication_params(c_in_tile_size, K_total_used) if c_in_tile_size > 0 else (1, 1)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)
    input_stacked_list: list[Optional[nl.NkiTensor]] = []
    valid_offsets: list[int] = []
    valid_sizes: list[int] = []
    valid_dh_starts: list[int] = []
    valid_dh_counts: list[int] = []
    valid_w_starts: list[int] = []
    valid_w_counts: list[int] = []
    use_strided_flags: list[bool] = []

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = stacked_input_bufs[k_outer_idx]

        # Compute the valid free-dim range for this K-outer tile
        valid_free_start, valid_free_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            used_k_positions,
            k_start,
            k_end,
        )

        if valid_free_start >= valid_free_end:
            input_stacked_list.append(None)
            valid_offsets.append(0)
            valid_sizes.append(0)
            valid_dh_starts.append(0)
            valid_dh_counts.append(0)
            valid_w_starts.append(0)
            valid_w_counts.append(0)
            use_strided_flags.append(False)
            continue

        valid_free_size = valid_free_end - valid_free_start
        first_valid_dh_idx_pre = valid_free_start // w_tile_size
        last_valid_dh_idx_pre = (valid_free_end - 1) // w_tile_size
        k_w_valid_start_pre = valid_free_start - first_valid_dh_idx_pre * w_tile_size
        k_w_valid_end_pre = valid_free_end - last_valid_dh_idx_pre * w_tile_size
        valid_dh_count_pre = last_valid_dh_idx_pre - first_valid_dh_idx_pre + 1
        has_w_padding = k_w_valid_start_pre > 0 or k_w_valid_end_pre < w_tile_size

        # Check if all K positions have the same W valid range
        all_k_same_w_range_pre = True
        if k_actual > 1 and valid_dh_count_pre > 1 and has_w_padding:
            for k_check_pre in range(k_actual):
                flat_k_pre = used_k_positions[k_start + k_check_pre]
                _, _, k_w_pre = _decompose_k_position(flat_k_pre, K_h, K_w)
                base_w_pre = w_start * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_start_pre = div_ceil(-base_w_pre, cfg.stride_w) if base_w_pre < 0 else 0
                last_w_pre = (w_end - 1) * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_end_pre = (
                    w_tile_size - div_ceil(last_w_pre - cfg.W + 1, cfg.stride_w) if last_w_pre >= cfg.W else w_tile_size
                )
                if this_w_start_pre != k_w_valid_start_pre or this_w_end_pre != k_w_valid_end_pre:
                    all_k_same_w_range_pre = False
                    break

        will_use_strided = all_k_same_w_range_pre and valid_dh_count_pre > 1 and has_w_padding

        # Check if any positions within the valid free-dim range need zero memset.
        needs_zero_init = partition_stride > c_in_tile_size
        if not needs_zero_init:
            first_valid_dh_idx = valid_free_start // w_tile_size
            last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
            for dh_idx in range(first_valid_dh_idx, last_valid_dh_idx + 1):
                d_out, h_out = dh_positions[dh_idx]
                # Determine what portion of this dh's W range falls within the valid range
                dh_free_start = dh_idx * w_tile_size
                dh_free_end = (dh_idx + 1) * w_tile_size
                # Clip to the valid range
                check_w_start = max(dh_free_start, valid_free_start) - dh_free_start
                check_w_end = min(dh_free_end, valid_free_end) - dh_free_start
                for k_rep_idx in range(k_actual):
                    flat_k_pos = used_k_positions[k_start + k_rep_idx]
                    k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < 0 or d_in >= cfg.D or h_in < 0 or h_in >= cfg.H:
                        needs_zero_init = True
                        break
                    # Check if W range within the valid portion has any padding
                    if not will_use_strided:
                        base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        first_valid_w = 0 if base_w_in >= 0 else div_ceil(-base_w_in, cfg.stride_w)
                        last_w_in = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        last_valid_w = (
                            w_tile_size
                            if last_w_in < cfg.W
                            else w_tile_size - div_ceil(last_w_in - cfg.W + 1, cfg.stride_w)
                        )
                        if first_valid_w > check_w_start or last_valid_w < check_w_end:
                            needs_zero_init = True
                            break
                if needs_zero_init:
                    break

        if needs_zero_init:
            # Only memset the valid range portion of the buffer
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        if input_window != None:
            for k_rep_idx in range(k_actual):
                partition_offset = k_rep_idx * partition_stride
                flat_k_pos = used_k_positions[k_start + k_rep_idx]
                k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left

                # Compute valid W range
                first_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_start - base_w_in, cfg.stride_w) if base_w_in < valid_w_start else 0,
                        w_tile_size,
                    ),
                )
                last_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_end - base_w_in, cfg.stride_w)
                        if base_w_in + (w_tile_size - 1) * cfg.stride_w >= valid_w_end
                        else w_tile_size,
                        w_tile_size,
                    ),
                )

                if first_valid_out >= last_valid_out:
                    continue

                src_w_start = base_w_in + first_valid_out * cfg.stride_w - valid_w_start
                num_w_elements = last_valid_out - first_valid_out
                src_w_end = src_w_start + (num_w_elements - 1) * cfg.stride_w + 1

                # Group consecutive dh_idx values to issue fewer, larger strided tensor_copy operations.
                dh_idx = 0
                while dh_idx < num_dh_stacked:
                    d_out, h_out = dh_positions[dh_idx]
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < valid_d_start or d_in >= valid_d_end or h_in < valid_h_start or h_in >= valid_h_end:
                        dh_idx += 1
                        continue

                    d_local_start = d_in - valid_d_start
                    h_local_start = h_in - valid_h_start
                    group_start_dh = dh_idx

                    # Group consecutive H positions with the same d_in
                    num_h = 1
                    next_dh = dh_idx + 1
                    while next_dh < num_dh_stacked:
                        d_out_next, h_out_next = dh_positions[next_dh]
                        d_in_next = d_out_next * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_next = h_out_next * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        if d_in_next != d_in:
                            break
                        if h_in_next < valid_h_start or h_in_next >= valid_h_end:
                            break
                        expected_h_in = h_in + num_h * cfg.stride_h
                        if h_in_next != expected_h_in:
                            break
                        num_h += 1
                        next_dh += 1

                    # Extend across D positions. Each D group must have exactly num_h consecutive H positions
                    # with the same h_local_start, and d_in must advance by stride_d.
                    num_d = 1
                    while next_dh + num_h - 1 < num_dh_stacked:
                        # Check if the next num_h positions form a valid D+H group
                        d_out_check, h_out_check = dh_positions[next_dh]
                        d_in_check = d_out_check * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_check = h_out_check * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        expected_d_in = d_in + num_d * cfg.stride_d
                        if d_in_check != expected_d_in:
                            break
                        if d_in_check < valid_d_start or d_in_check >= valid_d_end:
                            break
                        if h_in_check != h_in:
                            break
                        # Check if all num_h positions in this D group match
                        valid_d_group = True
                        for h_check_idx in range(1, num_h):
                            d_out_c, h_out_c = dh_positions[next_dh + h_check_idx]
                            d_in_c = d_out_c * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                            h_in_c = h_out_c * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                            if d_in_c != expected_d_in:
                                valid_d_group = False
                                break
                            expected_h_c = h_in + h_check_idx * cfg.stride_h
                            if h_in_c != expected_h_c:
                                valid_d_group = False
                                break
                            if h_in_c < valid_h_start or h_in_c >= valid_h_end:
                                valid_d_group = False
                                break
                        if not valid_d_group:
                            break
                        num_d += 1
                        next_dh += num_h

                    total_dh_in_group = num_d * num_h

                    if total_dh_in_group == 1:
                        # Single dh position, 1D strided copy
                        dst_start = group_start_dh * w_tile_size + first_valid_out
                        dst_end = dst_start + num_w_elements
                        nisa.tensor_copy(
                            dst=raw_buf[partition_offset : partition_offset + c_in_tile_size, dst_start:dst_end],
                            src=input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .select(dim=1, index=h_local_start)
                            .slice(dim=1, start=src_w_start, end=src_w_end, step=cfg.stride_w),
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    elif num_d == 1:
                        # Multiple H positions, 2D strided copy
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .slice(dim=1, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=2, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + num_h,
                                step=1,
                            )
                            .slice(dim=2, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    else:
                        # Multiple D and H positions, 3D strided copy
                        d_local_end = d_local_start + (num_d - 1) * cfg.stride_d + 1
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .slice(dim=1, start=d_local_start, end=d_local_end, step=cfg.stride_d)
                            .slice(dim=2, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=3, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + total_dh_in_group,
                                step=1,
                            )
                            .reshape_dim(1, (num_d, num_h))
                            .slice(dim=3, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )

                    dh_idx = next_dh

        # Compute per-K dh and W valid ranges for strided matmul
        # These are the same values used in _compute_valid_free_range
        first_valid_dh_idx, k_w_valid_start_val = divmod(valid_free_start, w_tile_size)
        last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
        k_w_valid_end_val = valid_free_end - last_valid_dh_idx * w_tile_size
        valid_dh_count = last_valid_dh_idx - first_valid_dh_idx + 1
        valid_w_count = k_w_valid_end_val - k_w_valid_start_val

        # Determine if strided matmul can be used: requires all K positions in the
        # tile to have the same W valid range. Note for K_REP=1 this is trivially true.
        all_k_same_w_range = True
        if k_actual > 1 and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size):
            ref_w_start_check = k_w_valid_start_val
            ref_w_end_check = k_w_valid_end_val
            for k_check_idx in range(k_actual):
                flat_k_check = used_k_positions[k_start + k_check_idx]
                _, _, k_w_check = _decompose_k_position(flat_k_check, K_h, K_w)
                base_w_check = w_start * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if base_w_check < 0:
                    this_w_start = div_ceil(-base_w_check, cfg.stride_w)
                else:
                    this_w_start = 0
                last_w_check = (w_end - 1) * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if last_w_check >= cfg.W:
                    this_w_end = w_tile_size - div_ceil(last_w_check - cfg.W + 1, cfg.stride_w)
                else:
                    this_w_end = w_tile_size
                if this_w_start != ref_w_start_check or this_w_end != ref_w_end_check:
                    all_k_same_w_range = False
                    break

        use_strided = (
            all_k_same_w_range and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size)
        )

        # For the strided matmul approach: create a strided view of the stacked buffer
        # that matches the PSUM strided view.
        if use_strided:
            # Strided case
            input_strided = (
                raw_buf.slice(dim=0, start=0, end=stacked_filter_dim, step=1)
                .slice(dim=1, start=0, end=effective_free_dim, step=1)
                .reshape_dim(1, (num_dh_stacked, w_tile_size))
                .slice(dim=1, start=first_valid_dh_idx, end=first_valid_dh_idx + valid_dh_count, step=1)
                .slice(dim=2, start=k_w_valid_start_val, end=k_w_valid_end_val, step=1)
            )
            input_stacked_list.append(input_strided)
        else:
            # Contiguous case
            input_stacked_list.append(raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end])

        valid_offsets.append(valid_free_start)
        valid_sizes.append(valid_free_size)
        valid_dh_starts.append(first_valid_dh_idx)
        valid_dh_counts.append(valid_dh_count)
        valid_w_starts.append(k_w_valid_start_val)
        valid_w_counts.append(valid_w_count)
        use_strided_flags.append(use_strided)
    return (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        valid_dh_starts,
        valid_dh_counts,
        valid_w_starts,
        valid_w_counts,
        use_strided_flags,
    )


def _conv3d_matmul(
    input_stacked_list: list[Optional[nl.NkiTensor]],
    filters_stacked_list: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    valid_offsets: list[int],
    valid_sizes: list[int],
    effective_free_dim: int,
    num_dh_stacked: int,
    w_tile_size: int,
    valid_dh_starts: list[int],
    valid_dh_counts: list[int],
    valid_w_starts: list[int],
    valid_w_counts: list[int],
    use_strided_flags: list[bool],
) -> None:
    """
    Accumulate matmul results across K-outer tiles into PSUM for all C_out tiles.

    Uses strided 2D views to handle W-padding without memsets: reshapes the free
    dimension as (num_dh, w_tile) and slices both the dh and W dimensions to
    exclude padded positions. This creates a strided access pattern where each
    dh row contributes only valid_w_count elements with stride w_tile_size,
    matching the flat compiler's approach.
    """
    for k_outer_idx in range(len(input_stacked_list)):
        input_stacked = input_stacked_list[k_outer_idx]
        if input_stacked == None:
            continue
        k_offset = valid_offsets[k_outer_idx]
        k_size = valid_sizes[k_outer_idx]
        dh_start = valid_dh_starts[k_outer_idx]
        dh_count = valid_dh_counts[k_outer_idx]
        w_start_k = valid_w_starts[k_outer_idx]
        w_count = valid_w_counts[k_outer_idx]
        for c_out_idx in range(len(psum_tiles)):
            if k_offset == 0 and k_size == effective_free_dim:
                # Full free dim
                nisa.nc_matmul(
                    dst=psum_tiles[c_out_idx],
                    stationary=filters_stacked_list[c_out_idx][k_outer_idx],
                    moving=input_stacked,
                )
            elif not use_strided_flags[k_outer_idx]:
                # Contiguous path (single dh, full W, or K_REP>1 with different W ranges)
                psum_sub = (psum_tiles[c_out_idx]).slice(dim=1, start=k_offset, end=k_offset + k_size, step=1)
                nisa.nc_matmul(
                    dst=psum_sub,
                    stationary=filters_stacked_list[c_out_idx][k_outer_idx],
                    moving=input_stacked,
                )
            else:
                # Multiple dh positions with W-padding and K_REP=1
                psum_sub = (
                    (psum_tiles[c_out_idx])
                    .slice(dim=1, start=0, end=effective_free_dim, step=1)
                    .reshape_dim(1, (num_dh_stacked, w_tile_size))
                    .slice(dim=1, start=dh_start, end=dh_start + dh_count, step=1)
                    .slice(dim=2, start=w_start_k, end=w_start_k + w_count, step=1)
                )
                nisa.nc_matmul(
                    dst=psum_sub,
                    stationary=filters_stacked_list[c_out_idx][k_outer_idx],
                    moving=input_stacked,
                )


def _conv3d_cin_tile(
    input_window: Optional[nl.NkiTensor],
    filters_for_cin: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    effective_free_dim: int,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
) -> None:
    """
    Process one C_in tile: scatter input into stacked layout and accumulate matmul into PSUM.

    Combines the scatter and matmul steps for a single input channel tile. First
    scatters the input window into K-replicated stacked buffers, then performs
    matmul accumulation across all C_out tiles.
    """
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        v_dh_starts,
        v_dh_counts,
        v_w_starts,
        v_w_counts,
        v_strided_flags,
    ) = _scatter_input_to_stacked(
        input_window=input_window,
        stacked_input_bufs=stacked_input_bufs,
        cfg=cfg,
        c_in_tile_size=c_in_tile_size,
        dh_positions=dh_positions,
        w_start=w_start,
        w_end=w_end,
        valid_d_start=valid_d_start,
        valid_d_end=valid_d_end,
        valid_h_start=valid_h_start,
        valid_h_end=valid_h_end,
        valid_w_start=valid_w_start,
        valid_w_end=valid_w_end,
        used_k_positions=used_k_positions,
        tensor_copy_engine_idx=tensor_copy_engine_idx,
        tensor_copy_engine_modulo=tensor_copy_engine_modulo,
        memset_engine_modulo=memset_engine_modulo,
    )
    _conv3d_matmul(
        input_stacked_list=input_stacked_list,
        filters_stacked_list=filters_for_cin,
        psum_tiles=psum_tiles,
        valid_offsets=valid_offsets,
        valid_sizes=valid_sizes,
        effective_free_dim=effective_free_dim,
        num_dh_stacked=num_dh_stacked,
        w_tile_size=w_tile_size,
        valid_dh_starts=v_dh_starts,
        valid_dh_counts=v_dh_counts,
        valid_w_starts=v_w_starts,
        valid_w_counts=v_w_counts,
        use_strided_flags=v_strided_flags,
    )


# Memory Hierarchy 2: HBM <-> SBUF interleaved with SBUF operations


def _apply_bias_activation_and_copy(
    psum_tiles: list[nl.NkiTensor],
    result_sbufs: list[nl.NkiTensor],
    bias_sbufs: list[Optional[nl.NkiTensor]],
    c_out_tile_sizes: list[int],
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
    effective_free_dim: int,
    unwritten_free_ranges: Optional[list[tuple[int, int]]] = None,
    memset_engine_modulo: int = 2,
) -> None:
    """
    Apply optional bias addition and activation, then copy PSUM results to SBUF.

    For each C_out tile, reads the accumulated PSUM result and applies bias
    addition and/or activation function as configured. If neither is needed,
    performs a direct tensor copy from PSUM to SBUF.
    """
    has_unwritten = unwritten_free_ranges != None and len(unwritten_free_ranges) > 0

    for c_out_tile_idx in range(len(result_sbufs)):
        psum_tile = psum_tiles[c_out_tile_idx]
        result_full = result_sbufs[c_out_tile_idx]
        bias_sbuf = bias_sbufs[c_out_tile_idx]
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        result = result_full[:c_out_tile_size, :effective_free_dim]

        if has_bias and has_activation:
            nisa.tensor_scalar(dst=result, data=psum_tile, op0=nl.add, operand0=bias_sbuf)
            nisa.activation(dst=result, data=result, op=get_nl_act_fn_from_type(activation_fn))
        elif has_bias:
            nisa.tensor_scalar(dst=result, data=psum_tile, op0=nl.add, operand0=bias_sbuf)
        elif has_activation:
            nisa.activation(dst=result, data=psum_tile, op=get_nl_act_fn_from_type(activation_fn))
        else:
            nisa.tensor_copy(dst=result, src=psum_tile)

        if not has_unwritten:
            continue

        # Handle never-written to PSUM locations f(0 + bias)
        for gap_idx in range(len(unwritten_free_ranges)):
            gap_start, gap_end = unwritten_free_ranges[gap_idx]
            gap_result = result_full[:c_out_tile_size, gap_start:gap_end]
            nisa.memset(
                dst=gap_result,
                value=0.0,
                engine=_get_memset_engine(c_out_tile_idx + gap_idx, memset_engine_modulo),
            )
            if has_bias and has_activation:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=bias_sbuf)
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))
            elif has_bias:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=bias_sbuf)
            elif has_activation:
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))


def _load_input_window_to_sbuf_3d(
    x_in_cin: nl.NkiTensor,
    input_window_buf: nl.NkiTensor,
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> tuple[Optional[nl.NkiTensor], int, int, int, int, int, int]:
    """
    Load the union input window from HBM into an SBUF buffer via DMA.

    Computes the bounding box of all receptive fields for the given D-H output
    positions and W tile range, clips to valid input bounds, and issues a single
    DMA copy. Returns None if the entire window falls in padding.
    """
    c_in_tile_size = x_in_cin.shape[0]

    union_d_start, union_d_end = cfg.D, 0
    union_h_start, union_h_end = cfg.H, 0
    for dh_idx in range(len(dh_positions)):
        d_out, h_out = dh_positions[dh_idx]
        field_d_start = d_out * cfg.stride_d - cfg.pad_d_left
        field_d_end = d_out * cfg.stride_d + (cfg.K_d - 1) * cfg.dilation_d - cfg.pad_d_left + 1
        field_h_start = h_out * cfg.stride_h - cfg.pad_h_top
        field_h_end = h_out * cfg.stride_h + (cfg.K_h - 1) * cfg.dilation_h - cfg.pad_h_top + 1
        union_d_start = min(union_d_start, field_d_start)
        union_d_end = max(union_d_end, field_d_end)
        union_h_start = min(union_h_start, field_h_start)
        union_h_end = max(union_h_end, field_h_end)

    field_w_start = w_start * cfg.stride_w - cfg.pad_w_left
    field_w_end = (w_end - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w - cfg.pad_w_left + 1

    valid_d_start, valid_d_end = max(0, union_d_start), min(cfg.D, union_d_end)
    valid_h_start, valid_h_end = max(0, union_h_start), min(cfg.H, union_h_end)
    valid_w_start, valid_w_end = max(0, field_w_start), min(cfg.W, field_w_end)

    if valid_d_start >= valid_d_end or valid_h_start >= valid_h_end or valid_w_start >= valid_w_end:
        return None, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end

    d_window_size = valid_d_end - valid_d_start
    h_window_size = valid_h_end - valid_h_start
    w_window_size = valid_w_end - valid_w_start

    nisa.dma_copy(
        dst=input_window_buf[:c_in_tile_size, :d_window_size, :h_window_size, :w_window_size],
        src=x_in_cin.slice(dim=1, start=valid_d_start, end=valid_d_end, step=1)
        .slice(dim=2, start=valid_h_start, end=valid_h_end, step=1)
        .slice(dim=3, start=valid_w_start, end=valid_w_end, step=1),
    )
    return input_window_buf, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end


def _conv3d_output_tile(
    x_in_batch: nl.NkiTensor,
    filters_cache: list[list[list[nl.NkiTensor]]],
    bias_cache: list[Optional[nl.NkiTensor]],
    result_sbufs: list[nl.NkiTensor],
    input_window_slots: list[nl.NkiTensor],
    stacked_input_slots: list[list[nl.NkiTensor]],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_tile_sizes: list[int],
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    psum_bank_idx: int,
    c_in_slot_offset: int = 0,
    preloaded_windows: Optional[list[tuple]] = None,
) -> None:
    """
    Compute one output tile by iterating over C_in tiles with multi-buffered I/O.

    When the number of c_out tiles exceeds NUM_PSUM_BANKS, processes them in
    sub-groups that fit within the available PSUM banks. For each sub-group:
    iterates over all C_in tiles (load input, scatter, matmul), then flushes
    PSUM results to SBUF before reusing the banks for the next sub-group.
    """
    num_c_out_tiles = len(result_sbufs)
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    effective_free_dim = num_dh_stacked * w_tile_size

    # Free-dim positions whose entire receptive field is padding receive no matmul.
    unwritten_free_ranges = _compute_unwritten_free_ranges(cfg, tile_cfg, dh_positions, w_start, w_end)

    # Process c_out tiles in sub-groups that fit within PSUM banks
    for sub_group_start in range(0, num_c_out_tiles, _NUM_PSUM_BANKS):
        sub_group_end = min(sub_group_start + _NUM_PSUM_BANKS, num_c_out_tiles)
        sub_group_size = sub_group_end - sub_group_start

        # Allocate PSUM banks for this sub-group
        psum_tiles = []
        for sg_idx in range(sub_group_size):
            c_out_tile_idx = sub_group_start + sg_idx
            psum_tiles.append(
                nl.ndarray(
                    shape=(c_out_tile_sizes[c_out_tile_idx], effective_free_dim),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, (psum_bank_idx + sg_idx) % _NUM_PSUM_BANKS * _PSUM_BANK_SIZE),
                )
            )

        # Iterate over all C_in tiles for this sub-group
        c_in_tile_idx = 0
        for c_in_start in range(0, cfg.C_in, tile_cfg.P_MAX):
            c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
            c_in_tile_size = c_in_end - c_in_start

            # Multi-buffer slot selection
            global_idx = c_in_slot_offset + c_in_tile_idx
            input_window_buf = input_window_slots[global_idx % len(input_window_slots)]
            stacked_input_bufs = stacked_input_slots[global_idx % len(stacked_input_slots)]

            if preloaded_windows != None:
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    preloaded_windows[c_in_tile_idx]
                )
            else:
                x_in_cin = x_in_batch.slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    _load_input_window_to_sbuf_3d(
                        x_in_cin=x_in_cin,
                        input_window_buf=input_window_buf,
                        cfg=cfg,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                    )
                )

            # Get filters for this sub-group
            filters_for_cin = []
            for sg_idx in range(sub_group_size):
                c_out_tile_idx = sub_group_start + sg_idx
                filters_for_cin.append(filters_cache[c_out_tile_idx][c_in_tile_idx])

            _conv3d_cin_tile(
                input_window=input_window,
                filters_for_cin=filters_for_cin,
                psum_tiles=psum_tiles,
                stacked_input_bufs=stacked_input_bufs,
                cfg=cfg,
                c_in_tile_size=c_in_tile_size,
                dh_positions=dh_positions,
                w_start=w_start,
                w_end=w_end,
                valid_d_start=valid_d_start,
                valid_d_end=valid_d_end,
                valid_h_start=valid_h_start,
                valid_h_end=valid_h_end,
                valid_w_start=valid_w_start,
                valid_w_end=valid_w_end,
                used_k_positions=tile_cfg.used_k_positions,
                tensor_copy_engine_idx=c_in_tile_idx * tile_cfg.K_total_used,
                effective_free_dim=effective_free_dim,
                tensor_copy_engine_modulo=tile_cfg.tensor_copy_activation_engine_offload,
                memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
            )
            c_in_tile_idx += 1

        # Flush this sub-group's PSUM results to SBUF
        sub_result_sbufs = result_sbufs[sub_group_start:sub_group_end]
        sub_bias_sbufs = bias_cache[sub_group_start:sub_group_end]
        sub_c_out_tile_sizes = c_out_tile_sizes[sub_group_start:sub_group_end]
        _apply_bias_activation_and_copy(
            psum_tiles=psum_tiles,
            result_sbufs=sub_result_sbufs,
            bias_sbufs=sub_bias_sbufs,
            c_out_tile_sizes=sub_c_out_tile_sizes,
            has_bias=cfg.has_bias,
            has_activation=cfg.has_activation,
            activation_fn=cfg.activation_fn,
            effective_free_dim=effective_free_dim,
            unwritten_free_ranges=unwritten_free_ranges,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
        )


# Memory Hierarchy 1: HBM <-> SBUF


def _allocate_filter_bias_buffers(
    sbm: SbufManager,
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    mem_cfg: Conv3dMemoryConfig,
    dtype,
) -> tuple[list[Optional[nl.NkiTensor]], list[list[nl.NkiTensor]]]:
    """
    Allocate SBUF heap buffers for bias and filter storage.

    Creates interleaved bias buffers (one per C_out tile in the interleave group)
    and filter buffers indexed by [c_in_tile][k_outer_tile].
    """
    c_out_wide = mem_cfg.c_out_interleave * tile_cfg.c_out_tile_size_max

    bias_bufs: list[Optional[nl.NkiTensor]] = []
    for tile_idx in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            bias_bufs.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.c_out_tile_size_max, 1),
                    dtype=nl.float32,
                    name=f"bias_{tile_idx}",
                )
            )
        else:
            bias_bufs.append(None)

    filter_bufs: list[list[nl.NkiTensor]] = []
    for cin_idx in range(tile_cfg.c_in_tile_count):
        cin_filters: list[nl.NkiTensor] = []
        for k_idx in range(tile_cfg.K_outer_tile_count):
            cin_filters.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, c_out_wide),
                    dtype=dtype,
                    name=f"filter_cin{cin_idx}_k{k_idx}",
                )
            )
        cin_filters_list = cin_filters
        filter_bufs.append(cin_filters_list)

    return bias_bufs, filter_bufs


def _load_bias_into_buf(
    bias: nl.NkiTensor,
    bias_buf: nl.NkiTensor,
    c_out_start: int,
    c_out_end: int,
) -> nl.NkiTensor:
    """Load a slice of the bias tensor from HBM into an SBUF buffer via DMA."""
    c_out_size = c_out_end - c_out_start
    nisa.dma_copy(
        dst=bias_buf[:c_out_size, 0],
        src=bias.slice(dim=0, start=c_out_start, end=c_out_end, step=1),
    )
    return bias_buf.slice(dim=0, start=0, end=c_out_size, step=1)


def _load_filters_into_bufs(
    filters: nl.NkiTensor,
    filter_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_start: int,
    c_in_end: int,
    c_out_group_start: int,
    c_out_group_end: int,
    used_k_positions: list[int],
    memset_engine_idx: int = 0,
    memset_engine_modulo: int = 2,
) -> None:
    """
    Load filter weights from HBM into K-replicated stacked SBUF buffers.

    For each K-outer tile, packs multiple filter positions along the partition
    dimension with the appropriate partition stride. Zero-initializes buffers
    when the partition stride exceeds the C_in tile size to handle padding.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    c_in_size = c_in_end - c_in_start
    c_out_wide = c_out_group_end - c_out_group_start
    K_REP, partition_stride = _get_k_replication_params(c_in_size, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = filter_bufs[k_outer_idx]

        if partition_stride > c_in_size:
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, :c_out_wide],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        for k_rep_idx in range(k_actual):
            partition_offset = k_rep_idx * partition_stride
            flat_k_pos = used_k_positions[k_start + k_rep_idx]
            k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
            nisa.dma_copy(
                dst=raw_buf[partition_offset : partition_offset + c_in_size, :c_out_wide],
                src=filters.select(dim=0, index=k_d_idx)
                .select(dim=0, index=k_h_idx)
                .select(dim=0, index=k_w_idx)
                .slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                .slice(dim=1, start=c_out_group_start, end=c_out_group_start + c_out_wide, step=1),
                dge_mode=nisa.dge_mode.hwdge,
            )


def _load_bias_and_filters_for_c_out_group(
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_group_start: int,
    c_out_group_end: int,
    pre_bias_bufs: list[Optional[nl.NkiTensor]],
    pre_filter_bufs: list[list[nl.NkiTensor]],
) -> tuple[list[int], list[Optional[nl.NkiTensor]], list[list[list[nl.NkiTensor]]]]:
    """
    Load bias and filter weights for an entire C_out interleave group.

    Loads bias per c_out tile, then loads all filter weights for the entire
    c_out group. Returns sliced views for each c_out tile.
    """
    actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)
    c_out_tile_sizes = []
    bias_cache: list[Optional[nl.NkiTensor]] = []

    for c_out_tile_idx in range(actual_group_size):
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = min(c_out_tile_start + tile_cfg.P_MAX, tile_cfg.C_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        c_out_tile_sizes.append(c_out_tile_size)

        if cfg.has_bias:
            bias_cache.append(
                _load_bias_into_buf(
                    bias,
                    pre_bias_bufs[c_out_tile_idx],
                    c_out_tile_start,
                    c_out_tile_end,
                )
            )
        else:
            bias_cache.append(None)

    K_total_used = len(tile_cfg.used_k_positions)
    K_REP, partition_stride = _get_k_replication_params(tile_cfg.c_in_tile_size_max, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)

    for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
        c_in_start = c_in_tile_idx * tile_cfg.P_MAX
        c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
        _load_filters_into_bufs(
            filters,
            pre_filter_bufs[c_in_tile_idx],
            cfg,
            c_in_start,
            c_in_end,
            c_out_group_start,
            c_out_group_end,
            used_k_positions=tile_cfg.used_k_positions,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
        )

    filters_cache: list[list[list[nl.NkiTensor]]] = []
    for c_out_tile_idx in range(actual_group_size):
        c_out_offset = c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        tile_filters: list[list[nl.NkiTensor]] = []
        for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
            c_in_size = min(tile_cfg.P_MAX, cfg.C_in - c_in_tile_idx * tile_cfg.P_MAX)
            K_REP_cin, partition_stride_cin = _get_k_replication_params(c_in_size, K_total_used)
            K_outer_tile_count_cin = div_ceil(K_total_used, K_REP_cin)
            cin_k_views: list[nl.NkiTensor] = []
            for k_outer_idx in range(K_outer_tile_count_cin):
                k_start = k_outer_idx * K_REP_cin
                k_end = min(k_start + K_REP_cin, K_total_used)
                k_actual = k_end - k_start
                stacked_filter_dim = partition_stride_cin * k_actual
                wide_buf = pre_filter_bufs[c_in_tile_idx][k_outer_idx]
                cin_k_views.append(wide_buf[:stacked_filter_dim, c_out_offset : c_out_offset + c_out_tile_size])
            tile_filters.append(cin_k_views)
        filters_cache.append(tile_filters)

    return c_out_tile_sizes, bias_cache, filters_cache
