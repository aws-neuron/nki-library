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

"""3D average pooling backward for NeuronCore using a separable spread of the gradient."""

from dataclasses import dataclass
from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from ...core.utils.logging import get_logger

_POOL_RANK = 3  # Number of pooled spatial axes (D, H, W)
_AP_NUM_MAX = 65535  # A hardware access-pattern element count is a uint16
_DTYPE_BYTES_2 = 2  # Byte width of a 2-byte element dtype (bf16/fp16)
_DTYPE_BYTES_4 = 4  # Byte width of a 4-byte element dtype (fp32)
_TRANSPOSE_CHUNK = 128  # Spatial elements per PE transpose
_SCALAR_COST_RATIO = 1.5  # Scalar-engine cost weight relative to Vector when balancing copy load
_PREFETCH_DEPTH = 6  # Number of grad_output work items to keep in flight


@nki.jit
def avg_pooling_3d_backward(
    grad_output: nl.NkiTensor,
    input_tensor: nl.NkiTensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    data_format: str = "NCDHW",
    output_format: Optional[str] = None,
) -> nl.NkiTensor:
    """Backward pass of separable 3D average pooling.

    Spreads each output gradient back over the input window it was averaged from, applying
    the 1/prod(kernel_size) scale, using a separable per-axis spread across D, H and W.

    Dimensions:
        B: Batch size
        C: Number of channels
        D: Input depth (spatial)
        H: Input height (spatial)
        W: Input width (spatial)
        Do: Output depth (pooled)
        Ho: Output height (pooled)
        Wo: Output width (pooled)

    Args:
        grad_output (nl.NkiTensor): Gradient w.r.t. the forward output, in HBM,
            output_format layout, forward-output shape.
        input_tensor (nl.NkiTensor): The forward input, data_format layout. Only its shape
            and dtype are read an average pool's gradient does not depend on the forward
            values.
        kernel_size (int or tuple): Pooling window (kD, kH, kW).
        stride (int or tuple, optional): Stride (sD, sH, sW); defaults to kernel_size.
        padding (int or tuple): Implicit zero padding (pD, pH, pW).
        data_format (str): Layout of input_tensor and grad_input, "NCDHW" or "NDHWC".
        output_format (str, optional): Layout of grad_output; defaults to data_format.

    Returns:
        grad_input (nl.NkiTensor): Gradient w.r.t. the forward input, in HBM,
        data_format layout, same spatial extents as input_tensor.

    Pseudocode:
        for tile of planes, tile of the input volume:
            g = grad_output over the outputs reaching this tile
            for axis in (W, H, D):
                # g[o*s + t] += g[o] for each tap, over the tile's padded window;
                # the first axis also applies the 1/prod(kernel_size) scale
                g = spread(g, axis)
            grad_input tile = g over the tile's owned box
    """
    cfg = _build_config(input_tensor, kernel_size, stride, padding, data_format, output_format)

    grad_input = _alloc_input(cfg)
    out_str = _layout_strides(cfg.output_format, cfg.C, cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2])
    in_str = _layout_strides(cfg.data_format, cfg.C, cfg.D, cfg.H, cfg.W)

    items = _work_items(cfg)
    if items:
        depth = min(cfg.prefetch_depth, len(items))
        loaded = []
        for pre_idx in range(depth):
            loaded.append(_load_out(grad_output, cfg, out_str, items[pre_idx][0], items[pre_idx][1], items[pre_idx][2]))
        for idx in range(len(items)):
            gout = loaded[idx]
            fetch_idx = idx + depth
            if fetch_idx < len(items):
                loaded.append(
                    _load_out(grad_output, cfg, out_str, items[fetch_idx][0], items[fetch_idx][1], items[fetch_idx][2])
                )
            part, plane_off, tile = items[idx][0], items[idx][1], items[idx][2]
            gin = _compute(gout, cfg, part, tile)
            _store(grad_input, cfg, in_str, gin, part, plane_off, tile)

    return grad_input


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
    """Coarsest (axis, chunk) cut whose tile satisfies fits."""
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
                    out_lo.append(o_lo)
                    out_lens.append(o_len)
                    own_off.append(own_lo[pool_axis] - (o_lo * st[pool_axis] - pd[pool_axis]))
                tiles.append((own_lo, own_lens, tuple(out_lo), tuple(out_lens), tuple(own_off)))
                w_lo += own_lens[2]
            h_lo += own_lens[1]
        d_lo += own_lens[0]
    return tiles


def _alloc_input(cfg):
    """Allocate the HBM grad_input in the requested data_format layout."""
    D, H, W = cfg.D, cfg.H, cfg.W
    if cfg.data_format == "NDHWC":
        return nl.ndarray((cfg.B, D, H, W, cfg.C), dtype=cfg.dtype, buffer=nl.shared_hbm)
    return nl.ndarray((cfg.B, cfg.C, D, H, W), dtype=cfg.dtype, buffer=nl.shared_hbm)


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
        chan = pos % C
        n = min(C - chan, end - pos)
        runs.append((pos - plane_off, pos // C, chan, n))
        pos += n
    return runs


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


def _load_out(grad_output, cfg, out_str, part, plane_off, tile):
    """Load the tile's swept (part, do, ho, wo) grad_output box."""
    out_lo, lens = tile[2], tile[3]
    out_vol = lens[0] * lens[1] * lens[2]
    s_b, s_c, s_d, s_h, s_w = out_str[0], out_str[1], out_str[2], out_str[3], out_str[4]
    base = out_lo[0] * s_d + out_lo[1] * s_h + out_lo[2] * s_w
    whole_plane = out_vol == cfg.out_spatial[0] * cfg.out_spatial[1] * cfg.out_spatial[2]

    box = nl.ndarray((part, lens[0], lens[1], lens[2]), dtype=cfg.dtype, buffer=nl.sbuf)

    if cfg.output_format == "NCDHW" and whole_plane and s_c == out_vol:
        src_view = grad_output.ap(pattern=[[s_c, part], [1, out_vol]], offset=plane_off * s_c)
        nisa.dma_copy(dst=box.reshape((part, out_vol)), src=src_view, dge_mode=nisa.dge_mode.none)
        return box

    runs = _batch_runs(plane_off, part, cfg.C)

    if cfg.output_format == "NCDHW":
        for run_idx in range(len(runs)):
            run = runs[run_idx]
            lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
            src_view = grad_output.ap(
                pattern=[[s_c, n], [s_d, lens[0]], [s_h, lens[1]], [s_w, lens[2]]],
                offset=batch_idx * s_b + chan * s_c + base,
            )
            nisa.dma_copy(dst=box[nl.ds(lane_lo, n), :, :, :], src=src_view, dge_mode=nisa.dge_mode.none)
        return box

    C = cfg.C
    dst_flat = box.reshape((part, out_vol))
    boxes = _box_runs(out_lo, lens, cfg.out_spatial)
    dst_off = 0
    for box_idx in range(len(boxes)):
        src_off, run_len = boxes[box_idx][0], boxes[box_idx][1]
        for run_idx in range(len(runs)):
            run = runs[run_idx]
            lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
            src_view = grad_output.ap(pattern=[[C, run_len], [1, n]], offset=batch_idx * s_b + src_off * C + chan)
            nisa.dma_transpose(dst=dst_flat[nl.ds(lane_lo, n), nl.ds(dst_off, run_len)], src=src_view, axes=(1, 0))
        dst_off += run_len
    return box


def _axis_plan(kernel_len, stride_len, out_len, own_len, own_off):
    """Per-tap writes of one axis over the owned extent, plus the first position to zero."""
    taps = []
    covered = [False] * own_len

    for tap in range(kernel_len):
        dst_start = tap - own_off
        lo = div_ceil(-dst_start, stride_len)
        if lo < 0:
            lo = 0
        hi = (own_len - 1 - dst_start) // stride_len
        if hi > out_len - 1:
            hi = out_len - 1
        if lo > hi:
            continue
        count = hi - lo + 1
        start = dst_start + lo * stride_len
        overwrite = tap < stride_len
        taps.append((lo, count, start, overwrite))
        if overwrite:
            for step_idx in range(count):
                covered[start + step_idx * stride_len] = True

    zero_from = own_len
    for pos in range(own_len):
        if not covered[pos]:
            zero_from = pos
            break
    return taps, zero_from


def _accum_volume(taps, part, num_out, num_in):
    """Element volume of the accumulating (Vector-only) taps of one axis."""
    total = 0
    for tap_idx in range(len(taps)):
        overwrite = taps[tap_idx][3]
        if not overwrite:
            total += part * num_out * taps[tap_idx][1] * num_in
    return total


def _copy_volumes(taps, part, num_out, num_in):
    """(volume, tap_idx) of the overwrite taps of one axis, largest volume first."""
    unsorted = []

    for tap_idx in range(len(taps)):
        if taps[tap_idx][3]:
            unsorted.append((part * num_out * taps[tap_idx][1] * num_in, tap_idx))

    n_vols = len(unsorted)
    taken = [False] * n_vols
    vols = []
    for _ in range(n_vols):
        best = -1
        for cand in range(n_vols):
            if taken[cand]:
                continue
            if best < 0:
                best = cand
            elif unsorted[cand][0] > unsorted[best][0]:
                best = cand
            elif unsorted[cand][0] == unsorted[best][0] and unsorted[cand][1] > unsorted[best][1]:
                best = cand
        taken[best] = True
        vols.append(unsorted[best])
    return vols


def _emit_spread(dst_view, src_view, scale, engine, accumulate):
    """Write (or add) scale * src_view into dst_view."""
    if accumulate:
        if scale == None:
            nisa.tensor_tensor(dst=dst_view, data1=dst_view, data2=src_view, op=nl.add)
        else:
            nisa.scalar_tensor_tensor(
                dst=dst_view,
                data=src_view,
                op0=nl.multiply,
                operand0=scale,
                op1=nl.add,
                operand1=dst_view,
            )
        return
    if scale == None:
        nisa.tensor_copy(dst=dst_view, src=src_view, engine=engine)
    else:
        nisa.tensor_scalar(dst=dst_view, data=src_view, op0=nl.multiply, operand0=scale, engine=engine)


def _fused_spread(dst, src, cfg, kernel_len, stride_len, out_len, own_len, own_off, part, num_out, num_in, scale):
    """Spread a whole axis in one instruction when the taps tile the owned extent exactly."""
    f_src = num_out * out_len * num_in
    f_dst = num_out * own_len * num_in
    src_pattern = [[f_src, part]]
    dst_pattern = [[f_dst, part]]
    if num_out > 1:
        src_pattern.append([out_len * num_in, num_out])
        dst_pattern.append([own_len * num_in, num_out])

    if out_len == 1:
        src_pattern.append([0, own_len])
        dst_pattern.append([num_in, own_len])
    else:
        src_pattern.append([num_in, out_len])
        dst_pattern.append([stride_len * num_in, out_len])
        src_pattern.append([0, stride_len])
        dst_pattern.append([num_in, stride_len])

    if num_in > 1:
        src_pattern.append([1, num_in])
        dst_pattern.append([1, num_in])

    _emit_spread(dst.ap(pattern=dst_pattern), src.ap(pattern=src_pattern), scale, cfg.copy_engine, False)


def _can_fuse_spread(kernel_len, stride_len, out_len, own_len, own_off, num_out, num_in):
    """Whether_fused_spread covers this axis exactly."""
    if out_len == 1:
        return own_off + own_len <= kernel_len
    if kernel_len != stride_len or own_off != 0 or own_len != out_len * stride_len:
        return False
    return num_out == 1 or num_in == 1


def _spread_axis(src, cfg, axis_idx, kernel_len, stride_len, out_len, own_len, own_off, part, scale, load):
    """Spread src along free-dim axis_idx onto that axis's owned extent."""
    dims = list(src.shape[1:])
    num_in = 1
    for dim_len in dims[axis_idx + 1 :]:
        num_in *= dim_len
    num_out = 1
    for dim_len in dims[:axis_idx]:
        num_out *= dim_len

    out_dims = list(dims)
    out_dims[axis_idx] = own_len
    dst = nl.ndarray((part, out_dims[0], out_dims[1], out_dims[2]), dtype=cfg.dtype, buffer=nl.sbuf)

    if _can_fuse_spread(kernel_len, stride_len, out_len, own_len, own_off, num_out, num_in):
        _fused_spread(dst, src, cfg, kernel_len, stride_len, out_len, own_len, own_off, part, num_out, num_in, scale)
        return dst, load

    f_src = num_out * out_len * num_in
    f_dst = num_out * own_len * num_in
    taps, zero_from = _axis_plan(kernel_len, stride_len, out_len, own_len, own_off)

    if zero_from < own_len:
        zero_pattern = [[f_dst, part]]
        if num_out > 1:
            zero_pattern.append([own_len * num_in, num_out])
        zero_pattern.append([num_in, own_len - zero_from])
        if num_in > 1:
            zero_pattern.append([1, num_in])
        nisa.memset(
            dst.ap(pattern=zero_pattern, offset=zero_from * num_in),
            value=0.0,
            engine=nisa.engine.gpsimd,
        )

    load[0] += _accum_volume(taps, part, num_out, num_in)
    copies = _copy_volumes(taps, part, num_out, num_in)
    engines = [None] * len(taps)

    for copy_idx in range(len(copies)):
        copy_pos = copies[copy_idx][0]
        tap_idx = copies[copy_idx][1]
        if load[0] <= load[1] * cfg.scalar_cost_ratio:
            engines[tap_idx] = nisa.engine.vector
            load[0] += copy_pos
        else:
            engines[tap_idx] = cfg.copy_engine
            load[1] += copy_pos

    for tap_idx in range(len(taps)):
        out_lo, count, dst_start, overwrite = taps[tap_idx]

        src_pattern = [[f_src, part]]
        if num_out > 1:
            src_pattern.append([out_len * num_in, num_out])
        src_pattern.append([num_in, count])
        if num_in > 1:
            src_pattern.append([1, num_in])
        src_view = src.ap(pattern=src_pattern, offset=out_lo * num_in)

        dst_pattern = [[f_dst, part]]
        if num_out > 1:
            dst_pattern.append([own_len * num_in, num_out])
        dst_pattern.append([stride_len * num_in, count])
        if num_in > 1:
            dst_pattern.append([1, num_in])
        dst_view = dst.ap(pattern=dst_pattern, offset=dst_start * num_in)

        _emit_spread(dst_view, src_view, scale, engines[tap_idx], not overwrite)
    return dst, load


def _compute(gout, cfg, part, tile):
    """Spread grad_output over the tile's owned box, normalising by 1/count on the way."""
    ks, st = cfg.ks, cfg.st
    own, out_lens, own_off = tile[1], tile[3], tile[4]

    count = 1
    for kernel_len in ks:
        count *= kernel_len
    scale = 1.0 / float(count)

    cur = gout
    pending = scale
    load = [0, 0]

    for axis_idx in range(_POOL_RANK - 1, -1, -1):
        if ks[axis_idx] == 1 and st[axis_idx] == 1 and out_lens[axis_idx] == own[axis_idx] and own_off[axis_idx] == 0:
            continue
        cur, load = _spread_axis(
            cur,
            cfg,
            axis_idx,
            ks[axis_idx],
            st[axis_idx],
            out_lens[axis_idx],
            own[axis_idx],
            own_off[axis_idx],
            part,
            pending,
            load,
        )
        pending = None

    if pending != None:
        scaled = nl.ndarray((part, own[0], own[1], own[2]), dtype=cfg.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=scaled[...],
            data=cur[...],
            op0=nl.multiply,
            operand0=pending,
            engine=cfg.copy_engine,
        )
        cur = scaled
    return cur


def _store(grad_input, cfg, in_str, gin, part, plane_off, tile):
    """Scatter the tile's owned gin box back to HBM in data_format layout."""
    own_lo, own = tile[0], tile[1]
    own_vol = own[0] * own[1] * own[2]
    s_b, s_c, s_d, s_h, s_w = in_str[0], in_str[1], in_str[2], in_str[3], in_str[4]
    src_tile = gin.reshape((part, own_vol))

    S = cfg.D * cfg.H * cfg.W
    base = own_lo[0] * s_d + own_lo[1] * s_h + own_lo[2] * s_w

    if cfg.data_format == "NCDHW" and own_vol == S and s_c == S:
        dst_view = grad_input.ap(pattern=[[s_c, part], [1, S]], offset=plane_off * s_c)
        nisa.dma_copy(dst=dst_view, src=src_tile, dge_mode=nisa.dge_mode.none)
        return

    runs = _batch_runs(plane_off, part, cfg.C)

    if cfg.data_format == "NCDHW":
        box = src_tile.reshape((part, own[0], own[1], own[2]))
        for run_idx in range(len(runs)):
            run = runs[run_idx]
            lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
            dst_view = grad_input.ap(
                pattern=[[s_c, n], [s_d, own[0]], [s_h, own[1]], [s_w, own[2]]],
                offset=batch_idx * s_b + chan * s_c + base,
            )
            nisa.dma_copy(dst=dst_view, src=box[nl.ds(lane_lo, n), :, :, :], dge_mode=nisa.dge_mode.none)
        return

    _store_ndhwc(grad_input, cfg, src_tile, part, plane_off, tile, runs, s_b)


def _store_ndhwc(grad_input, cfg, src_tile, part, plane_off, tile, runs, s_b):
    """PE-transpose the owned box to channels-last and store it in (spatial, C) order."""
    C = cfg.C
    own_lo, own = tile[0], tile[1]
    own_vol = own[0] * own[1] * own[2]
    boxes = _box_runs(own_lo, own, (cfg.D, cfg.H, cfg.W))
    merge_batches = (
        len(boxes) == 1
        and own_vol >= _TRANSPOSE_CHUNK
        and plane_off % C == 0
        and part % C == 0
        and boxes[0][0] % C == 0
    )

    if merge_batches:
        hbm_off = boxes[0][0]
        n_full = own_vol // _TRANSPOSE_CHUNK
        rem = own_vol - n_full * _TRANSPOSE_CHUNK
        n_batches = part // C
        base_batch = plane_off // C
        staged_all = nl.ndarray((_TRANSPOSE_CHUNK, n_full * part), dtype=cfg.dtype, buffer=nl.sbuf)
        for chunk_idx in range(n_full):
            transposed = nl.ndarray((_TRANSPOSE_CHUNK, part), dtype=cfg.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=transposed[:, :], data=src_tile[:, nl.ds(chunk_idx * _TRANSPOSE_CHUNK, _TRANSPOSE_CHUNK)]
            )
            nisa.tensor_copy(dst=staged_all[:, nl.ds(chunk_idx * part, part)], src=transposed[:, :])
        dst_view = grad_input.ap(
            pattern=[
                [C, _TRANSPOSE_CHUNK],
                [_TRANSPOSE_CHUNK * C, n_full],
                [s_b, n_batches],
                [1, C],
            ],
            offset=base_batch * s_b + hbm_off * C,
        )
        nisa.dma_copy(dst=dst_view, src=staged_all[:, :], dge_mode=nisa.dge_mode.none)
        if rem == 0:
            return
        _store_ndhwc_chunks(
            grad_input,
            cfg,
            src_tile,
            part,
            runs,
            s_b,
            [(hbm_off + n_full * _TRANSPOSE_CHUNK, rem)],
            n_full * _TRANSPOSE_CHUNK,
        )
        return

    _store_ndhwc_chunks(grad_input, cfg, src_tile, part, runs, s_b, boxes, 0)


def _store_ndhwc_chunks(grad_input, cfg, src_tile, part, runs, s_b, boxes, src_base):
    """Transpose and scatter each contiguous run of the owned box, one chunk at a time."""
    C = cfg.C
    src_off = src_base
    for box_idx in range(len(boxes)):
        hbm_off, run_len = boxes[box_idx][0], boxes[box_idx][1]
        sp = 0
        while sp < run_len:
            width = min(_TRANSPOSE_CHUNK, run_len - sp)
            transposed = nl.ndarray((_TRANSPOSE_CHUNK, part), dtype=cfg.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=transposed[0:width, :], data=src_tile[:, nl.ds(src_off + sp, width)])
            staged = nl.ndarray((_TRANSPOSE_CHUNK, part), dtype=cfg.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=staged[0:width, :], src=transposed[0:width, :])
            for run_idx in range(len(runs)):
                run = runs[run_idx]
                lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
                dst_view = grad_input.ap(
                    pattern=[[C, width], [1, n]],
                    offset=batch_idx * s_b + (hbm_off + sp) * C + chan,
                )
                nisa.dma_copy(dst=dst_view, src=staged[0:width, nl.ds(lane_lo, n)], dge_mode=nisa.dge_mode.none)
            sp += width
        src_off += run_len


def _per_axis(value, rank):
    """A scalar broadcasts to every spatial axis."""
    return [value] * rank if isinstance(value, int) else list(value)


def _out_dim(in_len, pad, kernel_len, stride_len):
    """Pooled output length of one axis."""
    return (in_len + 2 * pad - kernel_len) // stride_len + 1


def _is_2byte(dtype):
    """Whether the element dtype is 2 bytes wide."""
    return str(dtype) == str(nl.bfloat16) or str(dtype) == str(nl.float16)


def _tile_bytes(ks, st, own_lens, dtype_bytes, prefetch_depth):
    """Peak SBUF bytes per lane for one backward work item on an owned box of own_lens."""
    out_lens = []
    for axis_idx in range(_POOL_RANK):
        out_lens.append(_touching_count(own_lens[axis_idx], ks[axis_idx], st[axis_idx]))

    total = prefetch_depth * out_lens[0] * out_lens[1] * out_lens[2]
    total += out_lens[0] * out_lens[1] * own_lens[2]
    total += out_lens[0] * own_lens[1] * own_lens[2]
    total += own_lens[0] * own_lens[1] * own_lens[2]
    return total * dtype_bytes


def _tile_ap_ok(ks, st, own_lens):
    """Whether every access-pattern element count of this tile fits a uint16."""
    out_vol = 1
    for axis_idx in range(_POOL_RANK):
        out_vol *= _touching_count(own_lens[axis_idx], ks[axis_idx], st[axis_idx])
    own_vol = own_lens[0] * own_lens[1] * own_lens[2]
    return out_vol <= _AP_NUM_MAX and own_vol <= _AP_NUM_MAX


@dataclass(frozen=True)
class AvgPool3DBackwardConfig(nl.NKIObject):
    """Configuration for the 3D average-pool backward kernel."""

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

    # Tuning
    copy_engine: type
    scalar_cost_ratio: float
    prefetch_depth: int

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

    copy_engine = nisa.engine.scalar
    scalar_cost_ratio = _SCALAR_COST_RATIO
    prefetch_depth = _PREFETCH_DEPTH
    total_sbuf = nl.tile_size.total_available_sbuf_size
    dtype_bytes = _DTYPE_BYTES_2 if _is_2byte(input_tensor.dtype) else _DTYPE_BYTES_4

    def fits(own_lens):
        return (
            _tile_ap_ok(ks, st, own_lens) and _tile_bytes(ks, st, own_lens, dtype_bytes, prefetch_depth) <= total_sbuf
        )

    chunk_axis, chunk = _plan_chunk(in_spatial, fits)
    nominal = _tile_lens(in_spatial, chunk_axis, chunk)
    tiles = _grad_tiles(in_spatial, out_spatial, ks, st, pd, nominal)
    tiled = _is_tiled(in_spatial, nominal)
    item_bytes = _tile_bytes(ks, st, nominal, dtype_bytes, prefetch_depth)

    global_items = len(tiles) * div_ceil(B * C, partition_limit)
    local_items = div_ceil(global_items - prg_id, n_prgs) if global_items > prg_id else 0

    logger = get_logger("avg_pooling_3d_backward")
    logger.info(
        "AvgPool3DBackwardConfig: "
        "data_format=" + str(data_format) + ", "
        "output_format=" + str(output_format) + ", "
        "B=" + str(B) + ", C=" + str(C) + ", D=" + str(D) + ", H=" + str(H) + ", W=" + str(W) + ", "
        "kernel_size=" + str(ks) + ", stride=" + str(st) + ", padding=" + str(pd) + ", "
        "out_spatial=" + str(out_spatial) + ", dtype=" + str(input_tensor.dtype) + ", "
        "partition_limit=" + str(partition_limit) + ", "
        "chunk_axis=" + str(chunk_axis) + ", chunk=" + str(chunk) + ", "
        "grad_tiles=" + str(len(tiles)) + ", tiled=" + str(tiled) + ", "
        "tile_bytes=" + str(item_bytes) + "/" + str(total_sbuf) + ", "
        "scalar_cost_ratio=" + str(scalar_cost_ratio) + ", "
        "lnc(n_prgs)=" + str(n_prgs) + ", "
        "shard_id(prg_id)=" + str(prg_id) + ", "
        "global_work_items=" + str(global_items) + ", "
        "local_work_items=" + str(local_items)
    )

    return AvgPool3DBackwardConfig(
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
        copy_engine=copy_engine,
        scalar_cost_ratio=scalar_cost_ratio,
        prefetch_depth=prefetch_depth,
        n_prgs=n_prgs,
        prg_id=prg_id,
    )
