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

"""3D max pooling backward for NeuronCore using the input and output of max pooling 3D with a contribution mask strategy."""

from dataclasses import dataclass
from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from ...core.utils.logging import get_logger

_POOL_RANK = 3  # Number of pooled spatial axes (D, H, W)
_AP_NUM_MAX = 65535  # A hardware access-pattern element count is a uint16
_PARTITION_QUADRANTS = 4  # SBUF partition reads must start on a quadrant boundary
_TRANSPOSE_CHUNK = 128  # Spatial elements per PE transpose when de-interleaving channels
_DTYPE_BYTES_2 = 2  # Byte width of a 2-byte element dtype (bf16/fp16)
_DTYPE_BYTES_4 = 4  # Byte width of a 4-byte element dtype (fp32)
_PREFETCH_SETS = 2  # Tile sets live at once, so tile k+1's DMA overlaps tile k's compute
_MAX_WINDOW_TAPS = 4096  # Mask sweep emits a handful of instructions per window tap


@nki.jit
def max_pooling_3d_backward(
    grad_output: nl.NkiTensor,
    input_tensor: nl.NkiTensor,
    output_tensor: nl.NkiTensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    data_format: str = "NCDHW",
    output_format: Optional[str] = None,
) -> nl.NkiTensor:
    """Backward pass of separable 3D max pooling that uses the input and output of 3D max pooling forward and the backward gradient.

    Dimensions:
        B: batch.
        C: channels.
        D, H, W: the three pooled spatial axes.

    Args:
        grad_output (nl.NkiTensor): Gradient w.r.t. the forward output, in HBM,
            output_format layout, forward-output shape.
        input_tensor (nl.NkiTensor): The forward input, data_format layout.
        output_tensor (nl.NkiTensor): The forward output (per-window maxima),
            output_format layout, same shape as grad_output.
        kernel_size (int or tuple): Pooling window (kD, kH, kW) (int broadcasts).
        stride (int or tuple, optional): Stride (sD, sH, sW); defaults to kernel_size.
        padding (int or tuple): Implicit -inf padding (pD, pH, pW).
        data_format (str): Layout of input_tensor and grad_input, "NCDHW" or "NDHWC".
        output_format (str, optional): Layout of grad_output / output_tensor;
            defaults to data_format.

    Returns:
        grad_input (nl.NkiTensor): Gradient w.r.t. the forward input, in HBM,
        data_format layout, same spatial extents as input_tensor.

    Pseudocode:
        for tile of planes, tile of the input volume:
            inp = the input window; gout, omax = the outputs reaching this tile
            claimed = 0; gin = 0
            for (td, th, tw) in window taps (flattened order):
                in_view = inp at this tap, over the outputs it stays in bounds for
                eq      = (in_view == omax)
                new_win = (1 - claimed) * eq
                claimed += new_win
                gin_view += new_win * gout
            grad_input tile = gin over the tile's owned box
    """
    cfg = _build_config(input_tensor, kernel_size, stride, padding, data_format, output_format)

    grad_input = _alloc_input(cfg)
    in_str = _layout_strides(cfg.data_format, cfg.C, cfg.D, cfg.H, cfg.W)
    out_str = _layout_strides(cfg.output_format, cfg.C, cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2])

    items = _work_items(cfg)
    if items:
        nxt = _load_tile_set(input_tensor, output_tensor, grad_output, cfg, in_str, out_str, items[0])
        for idx in range(len(items)):
            cur = nxt
            if idx + 1 < len(items):
                nxt = _load_tile_set(input_tensor, output_tensor, grad_output, cfg, in_str, out_str, items[idx + 1])
            part, plane_off, tile = items[idx][0], items[idx][1], items[idx][2]
            gin = _alloc_gin(cfg, part, tile)
            _compute(gin, cur[0], cur[1], cur[2], cfg, part, tile)
            _store(grad_input, cfg, in_str, gin, part, plane_off, tile)

    return grad_input


def _window_span(out_lo, out_len, kernel_len, stride_len, pad_len, in_len):
    """The (in_lo, in_extent, pad_lo) input window of one axis's output range."""
    win_lo = out_lo * stride_len - pad_len
    win_hi = (out_lo + out_len - 1) * stride_len + kernel_len - pad_len
    lo = max(0, win_lo)
    hi = min(in_len, win_hi)
    return lo, max(0, hi - lo), lo - win_lo


def _tile_lens(extents, axis, chunk):
    """Per-axis tile extents when axis is chunked into chunk."""
    lens = []
    for pool_axis in range(_POOL_RANK):
        if pool_axis < axis:
            lens.append(1)
        elif pool_axis == axis:
            lens.append(min(chunk, extents[pool_axis]))
        else:
            lens.append(extents[pool_axis])
    return tuple(lens)


def _plan_chunk(extents, fits):
    """Coarsest (axis, chunk) cut whose tile satisfies fits.

    Cuts the most major axis that gets small enough, so minor axes stay whole (and DMAs
    stay long).  Returns the finest cut when nothing fits, for the caller to handle.
    """
    for axis in range(_POOL_RANK):
        if not fits(_tile_lens(extents, axis, 1)):
            continue
        lo, hi = 1, extents[axis]
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2
            if fits(_tile_lens(extents, axis, mid)):
                lo = mid
            else:
                hi = mid - 1
        return axis, div_ceil(extents[axis], div_ceil(extents[axis], lo))
    return _POOL_RANK - 1, 1


def _is_tiled(extents, lens):
    """Whether lens actually cuts extents into more than one tile."""
    for pool_axis in range(_POOL_RANK):
        if lens[pool_axis] < extents[pool_axis]:
            return True
    return False


def _flat_offset(pos, extents):
    """Row-major flat index of pos in a (D, H, W) volume of extents."""
    return (pos[0] * extents[1] + pos[1]) * extents[2] + pos[2]


def _box_runs(lo, lens, extents):
    """Maximal contiguous runs of a (D, H, W) box as (flat_off, run_len) pairs."""
    if lens[2] == extents[2] and lens[1] == extents[1]:
        return [(_flat_offset(lo, extents), lens[0] * lens[1] * lens[2])]
    if lens[2] == extents[2]:
        runs = []
        for d_idx in range(lens[0]):
            runs.append((_flat_offset((lo[0] + d_idx, lo[1], lo[2]), extents), lens[1] * lens[2]))
        return runs
    runs = []
    for d_idx in range(lens[0]):
        for h_idx in range(lens[1]):
            runs.append((_flat_offset((lo[0] + d_idx, lo[1] + h_idx, lo[2]), extents), lens[2]))
    return runs


def _touching_outputs(in_lo, in_len, kernel_len, stride_len, pad_len, out_len):
    """(lo, count) of the output positions whose window overlaps [in_lo, in_lo + in_len)."""
    lo = div_ceil(in_lo + pad_len - kernel_len + 1, stride_len)
    if lo < 0:
        lo = 0
    hi = (in_lo + in_len - 1 + pad_len) // stride_len
    if hi > out_len - 1:
        hi = out_len - 1
    return lo, max(0, hi - lo + 1)


def _touching_count(in_len, kernel_len, stride_len):
    """Upper bound on the outputs whose window overlaps an input run of in_len."""
    return (in_len + kernel_len - 2) // stride_len + 1


def _grad_tiles(in_spatial, out_spatial, ks, st, pd, lens):
    """Every input-space tile of the lens cut, in input order."""
    tiles = []
    d_lo = 0
    while d_lo < in_spatial[0]:
        h_lo = 0
        while h_lo < in_spatial[1]:
            w_lo = 0
            while w_lo < in_spatial[2]:
                own_lo = (d_lo, h_lo, w_lo)
                own_lens = (
                    min(lens[0], in_spatial[0] - d_lo),
                    min(lens[1], in_spatial[1] - h_lo),
                    min(lens[2], in_spatial[2] - w_lo),
                )
                out_lo = []
                out_lens = []
                data_lo = []
                data_lens = []
                own_off = []
                for pool_axis in range(_POOL_RANK):
                    o_lo, o_len = _touching_outputs(
                        own_lo[pool_axis],
                        own_lens[pool_axis],
                        ks[pool_axis],
                        st[pool_axis],
                        pd[pool_axis],
                        out_spatial[pool_axis],
                    )
                    lo, extent, _ = _window_span(
                        o_lo, o_len, ks[pool_axis], st[pool_axis], pd[pool_axis], in_spatial[pool_axis]
                    )
                    out_lo.append(o_lo)
                    out_lens.append(o_len)
                    data_lo.append(lo)
                    data_lens.append(extent)
                    own_off.append(own_lo[pool_axis] - lo)
                tiles.append(
                    (
                        own_lo,
                        own_lens,
                        tuple(out_lo),
                        tuple(out_lens),
                        tuple(data_lo),
                        tuple(data_lens),
                        tuple(own_off),
                    )
                )
                w_lo += own_lens[2]
            h_lo += own_lens[1]
        d_lo += own_lens[0]
    return tiles


def _alloc_gin(cfg, part, tile):
    """Allocate one fp32 gin accumulator over the tile's real input window."""
    data = tile[5]
    return nl.ndarray((part, data[0], data[1], data[2]), dtype=nl.float32, buffer=nl.sbuf)


def _load_tile_set(input_tensor, output_tensor, grad_output, cfg, in_str, out_str, item):
    """Load one tile's (inp, omax, gout) into a fresh buffer set for prefetching."""
    part, plane_off, tile = item[0], item[1], item[2]
    inp = _load_input(input_tensor, cfg, in_str, part, plane_off, tile)
    omax = _load_out(output_tensor, cfg, out_str, part, plane_off, tile)
    gout = _load_out(grad_output, cfg, out_str, part, plane_off, tile)
    return (inp, omax, gout)


def _alloc_input(cfg):
    """Allocate the HBM grad_input in the requested data_format layout."""
    D, H, W = cfg.D, cfg.H, cfg.W
    if cfg.data_format == "NDHWC":
        return nl.ndarray((cfg.B, D, H, W, cfg.C), dtype=cfg.dtype, buffer=nl.shared_hbm)
    return nl.ndarray((cfg.B, cfg.C, D, H, W), dtype=cfg.dtype, buffer=nl.shared_hbm)


def _work_items(cfg):
    """Return this program's (part, plane_off, tile) work items, round-robin over the list."""
    planes = cfg.B * cfg.C
    tiles = cfg.tiles
    items = []
    work_item_idx = 0
    plane_off = 0
    while plane_off < planes:
        part = min(cfg.partition_limit, planes - plane_off)
        for tile_idx in range(len(tiles)):
            if work_item_idx % cfg.n_prgs == cfg.prg_id:
                items.append((part, plane_off, tiles[tile_idx]))
            work_item_idx += 1
        plane_off += part
    return items


def _tap_plan(ks, st, pd, tile):
    """The in-bounds part of every window tap, plus whether gin needs pre-zeroing."""
    own_lens, out_lo, out_lens, data_lo, data_lens, own_off = tile[1], tile[2], tile[3], tile[4], tile[5], tile[6]
    taps = []
    seen = [False] * (data_lens[0] * data_lens[1] * data_lens[2])
    d_stride = data_lens[1] * data_lens[2]

    for td in range(ks[0]):
        for th in range(ks[1]):
            for tw in range(ks[2]):
                tap = (td, th, tw)
                in_off = []
                out_off = []
                cnt = []
                empty = False
                for axis_idx in range(_POOL_RANK):
                    lo, count, first = _tap_extent(
                        tap[axis_idx],
                        st[axis_idx],
                        pd[axis_idx],
                        out_lo[axis_idx],
                        out_lens[axis_idx],
                        data_lo[axis_idx],
                        data_lens[axis_idx],
                    )
                    if count <= 0:
                        empty = True
                    in_off.append(first)
                    out_off.append(lo)
                    cnt.append(count)
                if empty:
                    continue

                cells = []
                for d_idx in range(cnt[0]):
                    for h_idx in range(cnt[1]):
                        base = (in_off[0] + d_idx * st[0]) * d_stride + (in_off[1] + h_idx * st[1]) * data_lens[2]
                        for w_idx in range(cnt[2]):
                            cells.append(base + in_off[2] + w_idx * st[2])
                first_touch = True
                for cell_idx in range(len(cells)):
                    if seen[cells[cell_idx]]:
                        first_touch = False
                taps.append((tuple(in_off), tuple(out_off), tuple(cnt), first_touch))
                for cell_idx in range(len(cells)):
                    seen[cells[cell_idx]] = True

    # Explicit loops instead of a generator expression / unpacking comprehension:
    # NKI traces neither.
    for tap_idx in range(len(taps)):
        if not taps[tap_idx][3]:
            return taps, True

    for d_idx in range(own_lens[0]):
        for h_idx in range(own_lens[1]):
            for w_idx in range(own_lens[2]):
                cell = (own_off[0] + d_idx) * d_stride + (own_off[1] + h_idx) * data_lens[2] + (own_off[2] + w_idx)
                if not seen[cell]:
                    return taps, True
    return taps, False


def _tap_extent(tap, stride_len, pad_len, out_lo, out_len, data_lo, data_len):
    """(out offset, count, input offset) of one tap on one axis, clipped to real input."""
    lo = div_ceil(data_lo + pad_len - tap, stride_len)
    if lo < out_lo:
        lo = out_lo
    hi = (data_lo + data_len - 1 + pad_len - tap) // stride_len
    if hi > out_lo + out_len - 1:
        hi = out_lo + out_len - 1
    if lo > hi:
        return 0, 0, 0
    return lo - out_lo, hi - lo + 1, lo * stride_len + tap - pad_len - data_lo


def _load_box_runs(tensor, cfg, strides, part, plane_off, lo, lens, extents, dst_tile, contiguous_dst):
    """DMA a (D, H, W) box of one HBM tensor into dst_tile, one batch run at a time."""
    s_b, s_c, s_d, s_h, s_w = strides[0], strides[1], strides[2], strides[3], strides[4]
    base = lo[0] * s_d + lo[1] * s_h + lo[2] * s_w

    runs = _batch_runs(plane_off, part, cfg.C)
    for run_idx in range(len(runs)):
        run = runs[run_idx]
        lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
        src_view = tensor.ap(
            pattern=[[s_c, n], [s_d, lens[0]], [s_h, lens[1]], [s_w, lens[2]]],
            offset=batch_idx * s_b + chan * s_c + base,
        )
        if contiguous_dst:
            dst = dst_tile.reshape((part, lens[0], lens[1], lens[2]))[nl.ds(lane_lo, n), :, :, :]
        else:
            dst = dst_tile[nl.ds(lane_lo, n), :, :, :]
        nisa.dma_copy(dst=dst, src=src_view, dge_mode=nisa.dge_mode.none)


def _load_ndhwc_box(tensor, cfg, part, plane_off, lo, lens, extents, s_b):
    """Load a channels-last (D, H, W) box and PE-transpose it to (part, box volume)."""
    C = cfg.C
    vol = lens[0] * lens[1] * lens[2]
    out = nl.ndarray((part, vol), dtype=cfg.dtype, buffer=nl.sbuf)
    runs = _batch_runs(plane_off, part, C)
    boxes = _box_runs(lo, lens, extents)
    dst_off = 0
    for box_idx in range(len(boxes)):
        src_off, run_len = boxes[box_idx][0], boxes[box_idx][1]
        sp = 0
        while sp < run_len:
            csize = min(_TRANSPOSE_CHUNK, run_len - sp)
            stg = nl.ndarray((csize, part), dtype=cfg.dtype, buffer=nl.sbuf)
            for run_idx in range(len(runs)):
                run = runs[run_idx]
                lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
                src_view = tensor.ap(pattern=[[C, csize], [1, n]], offset=batch_idx * s_b + (src_off + sp) * C + chan)
                nisa.dma_copy(dst=stg[:, nl.ds(lane_lo, n)], src=src_view, dge_mode=nisa.dge_mode.none)
            psum_t = nl.ndarray((part, csize), dtype=cfg.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=psum_t, data=stg[...])
            nisa.tensor_copy(dst=out[:, nl.ds(dst_off + sp, csize)], src=psum_t, engine=cfg.copy_engine)
            sp += csize
        dst_off += run_len
    return out


def _load_input(input_tensor, cfg, in_str, part, plane_off, tile):
    """Load the tile's real (part, *data_lens) input window."""
    data_lo, lens = tile[4], tile[5]

    if cfg.data_format == "NDHWC":
        staged = _load_ndhwc_box(input_tensor, cfg, part, plane_off, data_lo, lens, (cfg.D, cfg.H, cfg.W), in_str[0])
        return staged.reshape((part, lens[0], lens[1], lens[2]))

    inp = nl.ndarray((part, lens[0], lens[1], lens[2]), dtype=cfg.dtype, buffer=nl.sbuf)
    _load_box_runs(input_tensor, cfg, in_str, part, plane_off, data_lo, lens, (cfg.D, cfg.H, cfg.W), inp, False)
    return inp


def _load_out(tensor, cfg, out_str, part, plane_off, tile):
    """Load the tile's swept output box (output_tensor or grad_output)."""
    out_extents = cfg.out_spatial
    out_lo, lens = tile[2], tile[3]

    if cfg.output_format == "NDHWC":
        staged = _load_ndhwc_box(tensor, cfg, part, plane_off, out_lo, lens, out_extents, out_str[0])
        return staged.reshape((part, lens[0], lens[1], lens[2]))

    box = nl.ndarray((part, lens[0], lens[1], lens[2]), dtype=cfg.dtype, buffer=nl.sbuf)
    _load_box_runs(tensor, cfg, out_str, part, plane_off, out_lo, lens, out_extents, box, False)
    return box


def _compute(gin, inp, omax, gout, cfg, part, tile):
    """Mask sweep over the window taps, accumulating into the caller's gin tile."""
    st = cfg.st
    out_lens, data = tile[3], tile[5]
    dt = cfg.dtype

    taps, needs_zero = _tap_plan(cfg.ks, st, cfg.pd, tile)
    if needs_zero:
        nisa.memset(gin[...], value=0.0, engine=nisa.engine.gpsimd)

    ug = nl.ndarray((part, out_lens[0], out_lens[1], out_lens[2]), dtype=dt, buffer=nl.sbuf)
    eq = nl.ndarray((part, out_lens[0], out_lens[1], out_lens[2]), dtype=dt, buffer=nl.sbuf)
    in_c = nl.ndarray((part, out_lens[0], out_lens[1], out_lens[2]), dtype=dt, buffer=nl.sbuf)
    new_win = nl.ndarray((part, out_lens[0], out_lens[1], out_lens[2]), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ug[...], src=gout[...], engine=cfg.copy_engine)

    in_free = data[0] * data[1] * data[2]
    out_free = out_lens[0] * out_lens[1] * out_lens[2]
    for tap_idx in range(len(taps)):
        in_off, out_off, cnt, first_touch = taps[tap_idx]

        in_pattern = [
            [in_free, part],
            [st[0] * data[1] * data[2], cnt[0]],
            [st[1] * data[2], cnt[1]],
            [st[2], cnt[2]],
        ]
        in_at = (in_off[0] * data[1] + in_off[1]) * data[2] + in_off[2]

        out_pattern = [
            [out_free, part],
            [out_lens[1] * out_lens[2], cnt[0]],
            [out_lens[2], cnt[1]],
            [1, cnt[2]],
        ]
        out_at = (out_off[0] * out_lens[1] + out_off[1]) * out_lens[2] + out_off[2]

        in_view = inp.ap(pattern=in_pattern, offset=in_at)
        gin_view = gin.ap(pattern=in_pattern, offset=in_at)
        in_c_view = in_c.ap(pattern=out_pattern, offset=out_at)
        eq_view = eq.ap(pattern=out_pattern, offset=out_at)
        ug_view = ug.ap(pattern=out_pattern, offset=out_at)
        new_win_view = new_win.ap(pattern=out_pattern, offset=out_at)
        omax_view = omax.ap(pattern=out_pattern, offset=out_at)

        nisa.tensor_copy(dst=in_c_view, src=in_view, engine=cfg.copy_engine)
        nisa.tensor_tensor(dst=eq_view, data1=in_c_view, data2=omax_view, op=nl.equal)

        nisa.tensor_tensor(dst=new_win_view, data1=eq_view, data2=ug_view, op=nl.multiply)
        nisa.tensor_tensor(dst=ug_view, data1=ug_view, data2=new_win_view, op=nl.subtract)
        if first_touch:
            nisa.tensor_copy(dst=gin_view, src=new_win_view, engine=cfg.copy_engine)
        else:
            nisa.tensor_tensor(dst=gin_view, data1=gin_view, data2=new_win_view, op=nl.add)

    return


def _store_ndhwc_run_contig(grad_input, cfg, in_str, gin, lane_lo, batch_idx):
    """Store one full-batch (C channels) run to NDHWC HBM in large contiguous bursts."""
    D, H, W, C = cfg.D, cfg.H, cfg.W, cfg.C
    DH = D * H
    WC = W * C
    DHW = D * H * W
    s_b = in_str[0]
    tile = cfg.tiles[0]
    data, own_off = tile[5], tile[6]
    data_free = data[0] * data[1] * data[2]

    interior = nl.ndarray((C, D, H, W), dtype=cfg.dtype, buffer=nl.sbuf)
    if data_free == DHW:
        nisa.tensor_copy(
            dst=interior.reshape((C, DHW))[...],
            src=gin.reshape((cfg.partition_limit, data_free)).ap(
                pattern=[[data_free, C], [1, DHW]], offset=lane_lo * data_free
            ),
            engine=cfg.copy_engine,
        )
    else:
        for dh in range(D * H):
            d_idx, h_idx = divmod(dh, H)
            nisa.tensor_copy(
                dst=interior.reshape((C, DHW)).ap(pattern=[[DHW, C], [1, W]], offset=dh * W),
                src=gin.reshape((cfg.partition_limit, data_free)).ap(
                    pattern=[[data_free, C], [1, W]],
                    offset=lane_lo * data_free
                    + (d_idx + own_off[0]) * data[1] * data[2]
                    + (h_idx + own_off[1]) * data[2]
                    + own_off[2],
                ),
                engine=cfg.copy_engine,
            )

    if DHW <= _TRANSPOSE_CHUNK:
        psum_t = nl.ndarray((DHW, C), dtype=cfg.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum_t, data=interior.reshape((C, DHW))[...])
        staging = nl.ndarray((DHW, C), dtype=cfg.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=staging[...], src=psum_t, engine=cfg.copy_engine)
        dst_view = grad_input.ap(pattern=[[C, DHW], [1, C]], offset=batch_idx * s_b)
        nisa.dma_copy(dst=dst_view, src=staging[...], dge_mode=nisa.dge_mode.none)
        return

    staging = nl.ndarray((DH, WC), dtype=cfg.dtype, buffer=nl.sbuf)
    w_group = max(1, min(W, 512 // C))
    w_off = 0
    while w_off < W:
        cur_group = min(w_group, W - w_off)
        psum_t = nl.ndarray((DH, cur_group * C), dtype=cfg.dtype, buffer=nl.psum)
        for col_idx in range(cur_group):
            nisa.nc_transpose(
                dst=psum_t[:, nl.ds(col_idx * C, C)],
                data=interior.reshape((C, DHW)).ap(pattern=[[DHW, C], [W, DH]], offset=w_off + col_idx),
            )
        nisa.tensor_copy(dst=staging[:, nl.ds(w_off * C, cur_group * C)], src=psum_t[...], engine=cfg.copy_engine)
        w_off += cur_group

    dst_view = grad_input.ap(pattern=[[WC, DH], [1, WC]], offset=batch_idx * s_b)
    nisa.dma_copy(dst=dst_view, src=staging[...], dge_mode=nisa.dge_mode.none)


def _store(grad_input, cfg, in_str, gin, part, plane_off, tile):
    """Scatter the tile's owned gin box back to HBM in data_format layout."""
    s_b, s_c, s_d, s_h, s_w = in_str[0], in_str[1], in_str[2], in_str[3], in_str[4]
    own_lo, own, data_lens, own_off = tile[0], tile[1], tile[5], tile[6]
    own_vol = own[0] * own[1] * own[2]

    if cfg.ndhwc_contig:
        runs = _batch_runs(plane_off, part, cfg.C)
        for run_idx in range(len(runs)):
            _store_ndhwc_run_contig(grad_input, cfg, in_str, gin, runs[run_idx][0], runs[run_idx][1])
        return

    whole_window = own_vol == data_lens[0] * data_lens[1] * data_lens[2]
    if whole_window:
        src_tile = gin.reshape((part, own_vol))
    else:
        staged = nl.ndarray((part, own_vol), dtype=cfg.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=staged.reshape((part, own[0], own[1], own[2]))[...],
            src=gin[
                :,
                nl.ds(own_off[0], own[0]),
                nl.ds(own_off[1], own[1]),
                nl.ds(own_off[2], own[2]),
            ],
            engine=cfg.copy_engine,
        )
        src_tile = staged

    base = own_lo[0] * s_d + own_lo[1] * s_h + own_lo[2] * s_w
    runs = _batch_runs(plane_off, part, cfg.C)
    box = src_tile.reshape((part, own[0], own[1], own[2]))
    for run_idx in range(len(runs)):
        run = runs[run_idx]
        lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
        dst_view = grad_input.ap(
            pattern=[[s_c, n], [s_d, own[0]], [s_h, own[1]], [s_w, own[2]]],
            offset=batch_idx * s_b + chan * s_c + base,
        )
        nisa.dma_copy(dst=dst_view, src=box[nl.ds(lane_lo, n), :, :, :], dge_mode=nisa.dge_mode.none)


def _layout_strides(fmt, C, D, H, W):
    """(batch, channel, d, h, w) element strides in HBM."""
    if fmt == "NDHWC":
        return (D * H * W * C, 1, H * W * C, W * C, C)
    return (C * D * H * W, D * H * W, H * W, W, 1)


def _batch_runs(plane_off, part, C):
    """Split lanes [plane_off, plane_off + part) into (lane_lo, batch, chan, n) runs."""
    runs = []
    pos = plane_off
    end = plane_off + part
    while pos < end:
        batch_idx, chan = divmod(pos, C)
        n = min(C - chan, end - pos)
        runs.append((pos - plane_off, batch_idx, chan, n))
        pos += n
    return runs


def _per_axis(value, rank):
    """A scalar broadcasts to every spatial axis."""
    return [value] * rank if isinstance(value, int) else list(value)


def _out_dim(in_len, pad, kernel_len, stride_len):
    """Pooled output length of one axis."""
    return (in_len + 2 * pad - kernel_len) // stride_len + 1


def _is_2byte(dtype):
    """Whether the element dtype is 2 bytes wide."""
    return str(dtype) == str(nl.bfloat16) or str(dtype) == str(nl.float16)


def _tile_bytes(ks, st, own_lens, dtype_bytes, stage_store):
    """Peak SBUF bytes per lane for one backward work item on an owned box of own_lens."""
    out_lens = []
    data = []
    for axis_idx in range(_POOL_RANK):
        count = _touching_count(own_lens[axis_idx], ks[axis_idx], st[axis_idx])
        out_lens.append(count)
        data.append((count - 1) * st[axis_idx] + ks[axis_idx])
    data_vol = data[0] * data[1] * data[2]
    out_vol = out_lens[0] * out_lens[1] * out_lens[2]
    own_vol = own_lens[0] * own_lens[1] * own_lens[2]

    total = _PREFETCH_SETS * dtype_bytes * (data_vol + 2 * out_vol)  # inp, omax, gout
    total += _DTYPE_BYTES_4 * data_vol  # gin
    total += dtype_bytes * 4 * out_vol  # ug, eq, in_c, new_win
    if stage_store:
        total += dtype_bytes * own_vol
    return total


def _tile_ap_ok(ks, st, own_lens):
    """Whether every access-pattern element count of this tile fits a uint16."""
    data_vol = 1
    out_vol = 1
    for axis_idx in range(_POOL_RANK):
        count = _touching_count(own_lens[axis_idx], ks[axis_idx], st[axis_idx])
        out_vol *= count
        data_vol *= (count - 1) * st[axis_idx] + ks[axis_idx]
    own_vol = own_lens[0] * own_lens[1] * own_lens[2]
    return data_vol <= _AP_NUM_MAX and out_vol <= _AP_NUM_MAX and own_vol <= _AP_NUM_MAX


@dataclass(frozen=True)
class MaxPool3DBackwardConfig(nl.NKIObject):
    """Configuration for the 3D max-pool backward kernel."""

    # Dimensions
    B: int
    C: int
    D: int
    H: int
    W: int
    ks: tuple
    st: tuple
    pd: tuple
    out_spatial: tuple
    dtype: type
    data_format: str
    output_format: str

    # Tiling
    partition_limit: int
    tiles: list
    ndhwc_contig: bool

    # Tuning
    copy_engine: type

    # Sharding
    n_prgs: int
    prg_id: int


def _build_config(input_tensor, kernel_size, stride, padding, data_format, output_format):
    """Parse inputs, derive tile geometry, and log the config."""
    if stride == None:
        stride = kernel_size
    if output_format == None:
        output_format = data_format

    partition_limit = nl.tile_size.pmax

    _, n_prgs, prg_id = get_program_sharding_info()

    ks = tuple(_per_axis(kernel_size, _POOL_RANK))
    st = tuple(_per_axis(stride, _POOL_RANK))
    pd = tuple(_per_axis(padding, _POOL_RANK))

    shape = list(input_tensor.shape)
    if data_format == "NDHWC":
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    else:
        B, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]

    in_spatial = (D, H, W)
    out_spatial = (
        _out_dim(D, pd[0], ks[0], st[0]),
        _out_dim(H, pd[1], ks[1], st[1]),
        _out_dim(W, pd[2], ks[2], st[2]),
    )

    n_taps = ks[0] * ks[1] * ks[2]
    kernel_assert(
        n_taps <= _MAX_WINDOW_TAPS,
        "max_pooling_3d_backward window is too large: prod(kernel_size)="
        + str(n_taps)
        + " exceeds "
        + str(_MAX_WINDOW_TAPS)
        + "; the mask sweep emits work per window position",
    )

    copy_engine = nisa.engine.scalar
    total_sbuf = nl.tile_size.total_available_sbuf_size
    dtype_bytes = _DTYPE_BYTES_2 if _is_2byte(input_tensor.dtype) else _DTYPE_BYTES_4

    def fits(own_lens):
        return _tile_ap_ok(ks, st, own_lens) and _tile_bytes(ks, st, own_lens, dtype_bytes, True) <= total_sbuf

    chunk_axis, chunk = _plan_chunk(in_spatial, fits)
    nominal = _tile_lens(in_spatial, chunk_axis, chunk)
    tiles = _grad_tiles(in_spatial, out_spatial, ks, st, pd, nominal)
    tiled = _is_tiled(in_spatial, nominal)
    item_bytes = _tile_bytes(ks, st, nominal, dtype_bytes, True)

    quadrant = partition_limit // _PARTITION_QUADRANTS
    ndhwc_contig = (
        not tiled
        and data_format == "NDHWC"
        and C <= partition_limit
        and D * H <= partition_limit
        and partition_limit % C == 0
        and C % quadrant == 0
    )

    global_items = len(tiles) * div_ceil(B * C, partition_limit)
    local_items = div_ceil(global_items - prg_id, n_prgs) if global_items > prg_id else 0

    logger = get_logger("max_pooling_3d_backward")
    logger.info(
        "MaxPool3DBackwardConfig: "
        "data_format=" + str(data_format) + ", "
        "output_format=" + str(output_format) + ", "
        "B=" + str(B) + ", C=" + str(C) + ", D=" + str(D) + ", H=" + str(H) + ", W=" + str(W) + ", "
        "kernel_size=" + str(ks) + ", stride=" + str(st) + ", padding=" + str(pd) + ", "
        "out_spatial=" + str(out_spatial) + ", dtype=" + str(input_tensor.dtype) + ", "
        "partition_limit=" + str(partition_limit) + ", "
        "chunk_axis=" + str(chunk_axis) + ", chunk=" + str(chunk) + ", "
        "grad_tiles=" + str(len(tiles)) + ", "
        "tile_bytes=" + str(item_bytes) + "/" + str(total_sbuf) + ", "
        "ndhwc_contig=" + str(ndhwc_contig) + ", "
        "lnc(n_prgs)=" + str(n_prgs) + ", "
        "shard_id(prg_id)=" + str(prg_id) + ", "
        "global_work_items=" + str(global_items) + ", "
        "local_work_items=" + str(local_items)
    )

    return MaxPool3DBackwardConfig(
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        ks=ks,
        st=st,
        pd=pd,
        out_spatial=out_spatial,
        dtype=input_tensor.dtype,
        data_format=data_format,
        output_format=output_format,
        partition_limit=partition_limit,
        tiles=tiles,
        ndhwc_contig=ndhwc_contig,
        copy_engine=copy_engine,
        n_prgs=n_prgs,
        prg_id=prg_id,
    )
