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

"""3D average pooling for NeuronCore using tensor_reduce on the Vector Engine."""

from dataclasses import dataclass
from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from ...core.utils.logging import get_logger

_INIT_VALUE = 0.0  # Fill value for padding (add identity)
_POOL_RANK = 3  # Number of pooled spatial axes (D, H, W)
_AP_NUM_MAX = 65535  # A hardware access-pattern element count is a uint16
_DTYPE_BYTES_2 = 2  # Byte width of a 2-byte element dtype (bf16/fp16)
_DTYPE_BYTES_4 = 4  # Byte width of a 4-byte element dtype (fp32)
_TRANSPOSE_ALIGN_BYTES = 32  # A transposing DMA's destination offset must be 32B aligned
_GROUP_BLOCK = "block"  # plane_off + grp_idx * part: generic, one DMA per group
_GROUP_LANE = "lane"  # plane_off + grp_idx: planes fully contiguous
_GROUP_BATCH = "batch"  # batch (b0 + grp_idx), fixed channel window: uniform HBM stride


@nki.jit
def avg_pooling_3d(
    src_tensor: nl.NkiTensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    data_format: str = "NCDHW",
    output_format: Optional[str] = None,
) -> nl.NkiTensor:
    """3D average pool kernel over the (D, H, W) axes. Pooling runs independently per (batch, channel).

    Dimensions:
        B: batch.
        C: channels.
        D, H, W: the three pooled spatial axes.

    Args:
        src_tensor (nl.NkiTensor): Input in HBM.  Shape is (B, C, D, H, W) for
            data_format="NCDHW" (channels-first) or (B, D, H, W, C) for
            data_format="NDHWC" (channels-last).
        kernel_size (int or tuple): Pooling window (kD, kH, kW) (int broadcasts).
        stride (int or tuple, optional): Stride (sD, sH, sW); defaults to
            kernel_size.
        padding (int or tuple): Implicit zero padding (pD, pH, pW).
        data_format (str): Input layout, "NCDHW" (default) or "NDHWC".
        output_format (str, optional): Output layout, "NCDHW" or "NDHWC";
            defaults to data_format (output matches input).

    Returns:
        output (nl.NkiTensor): Pooled result in HBM, in output_format layout:
        (B, C, D_out, H_out, W_out) or (B, D_out, H_out, W_out, C) with
        out[a] = (in[a] + 2*pad[a] - kernel[a]) // stride[a] + 1.

    Pseudocode:
        for (lane group, spatial tile) in work_items:
            cur = load_window(src_tensor, lane group, tile)
            res = reduce_all_axes(cur, tile) * 1/prod(kernel_size)
            store_window(out, res, lane group, tile)
        return out
    """
    cfg = _build_config(src_tensor, kernel_size, stride, padding, data_format, output_format)

    out = _alloc_output(cfg)

    in_str = _layout_strides(cfg.data_format, cfg.C, cfg.D, cfg.H, cfg.W)
    out_str = _layout_strides(cfg.output_format, cfg.C, cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2])

    items = _work_items(cfg)
    for item_idx in range(len(items)):
        item = items[item_idx]
        part = item[0]
        grp = item[1]
        plane_off = item[2]
        tile = item[3]

        if cfg.group_mode == _GROUP_BATCH:
            _batch_packed_chunked(src_tensor, out, cfg, part, grp, plane_off)
            continue

        if cfg.split_piece > 0:
            res = _pieced_sum(src_tensor, cfg, in_str, part, grp, plane_off, tile)
        else:
            out_lens, in_lo, in_lens, pad_lo = tile[1], tile[2], tile[3], tile[4]
            win = _win_lens(out_lens, cfg.ks, cfg.st)
            cur = _load_tile(src_tensor, cfg, in_str, part, grp, plane_off, win, in_lo, in_lens, pad_lo)
            res = _reduce_all_axes(cur, cfg, grp, out_lens, cfg.ks, cfg.st, cfg.scale)
        _store_tile(out, cfg, out_str, res, part, grp, plane_off, tile)

    return out


def _window_span(out_lo, out_len, kernel_len, stride_len, pad_len, in_len):
    """The (in_lo, in_extent, pad_lo) input window of one axis's output range."""
    win_lo = out_lo * stride_len - pad_len
    win_hi = (out_lo + out_len - 1) * stride_len + kernel_len - pad_len
    lo = max(0, win_lo)
    hi = min(in_len, win_hi)
    return lo, max(0, hi - lo), lo - win_lo


def _win_lens(out_lens, ks, st):
    """Padded window extent of a tile on each pooled axis."""
    lens = []
    for pool_axis in range(_POOL_RANK):
        lens.append((out_lens[pool_axis] - 1) * st[pool_axis] + ks[pool_axis])
    return tuple(lens)


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


def _spatial_tiles(out_spatial, in_spatial, ks, st, pd, lens):
    """Every tile of the lens cut, in output order."""
    tiles = []
    d_lo = 0
    while d_lo < out_spatial[0]:
        h_lo = 0
        while h_lo < out_spatial[1]:
            w_lo = 0
            while w_lo < out_spatial[2]:
                out_lo = (d_lo, h_lo, w_lo)
                out_lens = (
                    min(lens[0], out_spatial[0] - d_lo),
                    min(lens[1], out_spatial[1] - h_lo),
                    min(lens[2], out_spatial[2] - w_lo),
                )
                in_lo = []
                in_lens = []
                pad_lo = []
                for pool_axis in range(_POOL_RANK):
                    lo, extent, pad = _window_span(
                        out_lo[pool_axis],
                        out_lens[pool_axis],
                        ks[pool_axis],
                        st[pool_axis],
                        pd[pool_axis],
                        in_spatial[pool_axis],
                    )
                    in_lo.append(lo)
                    in_lens.append(extent)
                    pad_lo.append(pad)
                tiles.append((out_lo, out_lens, tuple(in_lo), tuple(in_lens), tuple(pad_lo)))
                w_lo += out_lens[2]
            h_lo += out_lens[1]
        d_lo += out_lens[0]
    return tiles


def _window_pieces(cfg, tile):
    """Split one tile's window along cfg.split_axis into separately summed pieces."""
    axis = cfg.split_axis
    tile_in_lo, tile_in_lens = tile[2], tile[3]
    pieces = []
    off = 0
    while off < tile_in_lens[axis]:
        piece = min(cfg.split_piece, tile_in_lens[axis] - off)
        in_lo = []
        in_lens = []
        kernel = []
        for pool_axis in range(_POOL_RANK):
            if pool_axis == axis:
                in_lo.append(tile_in_lo[axis] + off)
                in_lens.append(piece)
                kernel.append(piece)
            else:
                in_lo.append(tile_in_lo[pool_axis])
                in_lens.append(tile_in_lens[pool_axis])
                kernel.append(cfg.ks[pool_axis])
        pieces.append((tuple(in_lo), tuple(in_lens), tuple(kernel)))
        off += piece
    return pieces


def _pieced_sum(src_tensor, cfg, in_str, part, grp, plane_off, tile):
    """Sum one tile's window piece by piece, then scale to the mean."""
    out_lens = tile[1]
    acc = _alloc_tile(part, grp, out_lens[0], out_lens[1], out_lens[2], nl.float32)
    pieces = _window_pieces(cfg, tile)
    for piece_idx in range(len(pieces)):
        in_lo, in_lens, kernel = pieces[piece_idx][0], pieces[piece_idx][1], pieces[piece_idx][2]
        cur = _load_tile(src_tensor, cfg, in_str, part, grp, plane_off, kernel, in_lo, in_lens, cfg.no_pad)
        partial = _reduce_all_axes(cur, cfg, grp, out_lens, kernel, kernel, None)
        if piece_idx == 0:
            nisa.tensor_copy(dst=acc[...], src=partial[...], engine=cfg.copy_engine)
        else:
            nisa.tensor_tensor(dst=acc[...], data1=acc[...], data2=partial[...], op=nl.add)

    res = _alloc_tile(part, grp, out_lens[0], out_lens[1], out_lens[2], cfg.dtype)
    n_free = out_lens[0] * out_lens[1] * out_lens[2] * (grp if grp > 1 else 1)
    flat_pattern = [[n_free, part], [1, n_free]]
    nisa.tensor_scalar(
        dst=res.ap(pattern=flat_pattern),
        data=acc.ap(pattern=flat_pattern),
        op0=nl.multiply,
        operand0=cfg.scale,
        engine=cfg.copy_engine,
    )
    return res


def _batch_packed_chunked(src_tensor, out, cfg, part, grp, plane_off):
    """Fused load + reduce + store for a batch-packed global-pool tile."""
    S = cfg.D * cfg.H * cfg.W
    C = cfg.C
    b0 = plane_off // C
    chan = plane_off % C
    n_chunks = min(cfg.batch_chunks, grp)
    chunk = div_ceil(grp, n_chunks)

    res = nl.ndarray((part, grp), dtype=cfg.dtype, buffer=nl.sbuf)
    g0 = 0
    while g0 < grp:
        gc = min(chunk, grp - g0)
        staged = nl.ndarray((part, gc * S), dtype=cfg.dtype, buffer=nl.sbuf)
        if cfg.data_format == "NDHWC":
            if cfg.in_transpose:
                nisa.dma_transpose(
                    src=src_tensor.ap(pattern=[[C, gc * S], [1, part]], offset=(b0 + g0) * S * C + chan),
                    dst=staged,
                    axes=(1, 0),
                )
            else:
                nisa.dma_copy(
                    dst=staged.ap(pattern=[[gc * S, part], [1, gc * S]]),
                    src=src_tensor.ap(pattern=[[1, part], [S * C, gc], [C, S]], offset=(b0 + g0) * S * C + chan),
                    dge_mode=nisa.dge_mode.none,
                )
        else:
            nisa.dma_copy(
                dst=staged.ap(pattern=[[gc * S, part], [1, gc * S]]),
                src=src_tensor.ap(pattern=[[S, part], [C * S, gc], [1, S]], offset=(b0 + g0) * C * S + chan * S),
                dge_mode=nisa.dge_mode.none,
            )
        acc = nl.ndarray((part, gc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=acc[...], op=nl.add, data=staged.reshape((part, gc, S))[...], axis=2)
        nisa.tensor_scalar(
            dst=res[:, nl.ds(g0, gc)],
            data=acc[...],
            op0=nl.multiply,
            operand0=cfg.scale,
            engine=nisa.engine.scalar,
        )
        g0 += gc

    if cfg.two_byte:
        staged_out = nl.ndarray((grp, part), dtype=cfg.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(src=res, dst=staged_out, axes=(1, 0))
        nisa.dma_copy(
            dst=out.ap(pattern=[[C, grp], [1, part]], offset=b0 * C + chan),
            src=staged_out,
            dge_mode=nisa.dge_mode.none,
        )
    else:
        nisa.dma_copy(
            dst=out.ap(pattern=[[1, part], [C, grp]], offset=b0 * C + chan),
            src=res.ap(pattern=[[grp, part], [1, grp]]),
            dge_mode=nisa.dge_mode.none,
        )


def _alloc_output(cfg):
    """Allocate the HBM output in the requested layout."""
    Do, Ho, Wo = cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2]
    if cfg.output_format == "NDHWC":
        return nl.ndarray((cfg.B, Do, Ho, Wo, cfg.C), dtype=cfg.dtype, buffer=nl.shared_hbm)
    return nl.ndarray((cfg.B, cfg.C, Do, Ho, Wo), dtype=cfg.dtype, buffer=nl.shared_hbm)


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


def _group_plane(plane_off, part, grp_idx, cfg):
    """First plane of group grp_idx under the active group-to-plane mapping."""
    if cfg.group_mode == _GROUP_LANE:
        return plane_off + grp_idx
    if cfg.group_mode == _GROUP_BATCH:
        return plane_off + grp_idx * cfg.C
    return plane_off + grp_idx * part


def _align_up(value, multiple):
    """Round value up to the next multiple."""
    return div_ceil(value, multiple) * multiple


def _alloc_tile(part, grp, d0, d1, d2, dtype):
    """Allocate (part, [grp,] d0, d1, d2); the group dim is omitted when grp == 1."""
    if grp > 1:
        return nl.ndarray((part, grp, d0, d1, d2), dtype=dtype, buffer=nl.sbuf)
    return nl.ndarray((part, d0, d1, d2), dtype=dtype, buffer=nl.sbuf)


def _as_tile(flat, part, grp, d0, d1, d2):
    """View a flat (part, grp*d0*d1*d2) buffer as the canonical tile shape."""
    if grp > 1:
        return flat.reshape((part, grp, d0, d1, d2))
    return flat.reshape((part, d0, d1, d2))


def _load_tile(src_tensor, cfg, in_str, part, grp, plane_off, win, in_lo, in_lens, pad_lo):
    """Assemble the zero-padded (part, [grp,] *win) window of this lane group."""
    D, H, W = cfg.D, cfg.H, cfg.W
    s_b, s_c, s_d, s_h, s_w = in_str[0], in_str[1], in_str[2], in_str[3], in_str[4]
    vol = win[0] * win[1] * win[2]
    full_plane = in_lens[0] == D and in_lens[1] == H and in_lens[2] == W

    if cfg.in_transpose and full_plane:
        return _load_tile_transposed(src_tensor, cfg, part, grp, plane_off, win, pad_lo, s_b)

    cur = _alloc_tile(part, grp, win[0], win[1], win[2], cfg.dtype)

    if cfg.group_mode == _GROUP_LANE and grp > 1 and full_plane and vol == D * H * W:
        S = D * H * W
        nisa.dma_copy(
            dst=cur.reshape((part, grp * S)),
            src=src_tensor.ap(pattern=[[grp * S, part], [1, grp * S]], offset=plane_off * s_c),
            dge_mode=nisa.dge_mode.none,
        )
        return cur

    if in_lens[0] != win[0] or in_lens[1] != win[1] or in_lens[2] != win[2]:
        nisa.memset(cur[...], value=_INIT_VALUE, engine=nisa.engine.gpsimd)

    base = in_lo[0] * s_d + in_lo[1] * s_h + in_lo[2] * s_w
    cur_flat = cur.reshape((part, grp * vol)) if grp > 1 else cur
    for grp_idx in range(grp):
        runs = _batch_runs(_group_plane(plane_off, part, grp_idx, cfg), part, cfg.C)
        for run_idx in range(len(runs)):
            run = runs[run_idx]
            lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
            src_view = src_tensor.ap(
                pattern=[[s_c, n], [s_d, in_lens[0]], [s_h, in_lens[1]], [s_w, in_lens[2]]],
                offset=batch_idx * s_b + chan * s_c + base,
            )
            if grp > 1:
                dst_view = cur_flat.ap(
                    pattern=[[grp * vol, n], [win[1] * win[2], in_lens[0]], [win[2], in_lens[1]], [1, in_lens[2]]],
                    offset=lane_lo * grp * vol + grp_idx * vol + (pad_lo[0] * win[1] + pad_lo[1]) * win[2] + pad_lo[2],
                )
            else:
                dst_view = cur[
                    nl.ds(lane_lo, n),
                    nl.ds(pad_lo[0], in_lens[0]),
                    nl.ds(pad_lo[1], in_lens[1]),
                    nl.ds(pad_lo[2], in_lens[2]),
                ]
            nisa.dma_copy(dst=dst_view, src=src_view, dge_mode=nisa.dge_mode.none)
    return cur


def _load_tile_transposed(src_tensor, cfg, part, grp, plane_off, win, pad_lo, s_b):
    """Load whole channels-last planes by transposing (spatial, C) blocks into lanes."""
    D, H, W = cfg.D, cfg.H, cfg.W
    S = D * H * W
    slot = _align_up(S, cfg.transpose_align_elems)
    vol = win[0] * win[1] * win[2]
    flat = nl.ndarray((part, grp * slot), dtype=cfg.dtype, buffer=nl.sbuf)
    for grp_idx in range(grp):
        runs = _batch_runs(_group_plane(plane_off, part, grp_idx, cfg), part, cfg.C)
        s_off = 0
        while s_off < S:
            s_len = min(cfg.partition_limit, S - s_off)
            staged = nl.ndarray((s_len, part), dtype=cfg.dtype, buffer=nl.sbuf)
            for run_idx in range(len(runs)):
                run = runs[run_idx]
                lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
                src_view = src_tensor.ap(
                    pattern=[[cfg.C, s_len], [1, n]],
                    offset=batch_idx * s_b + chan + s_off * cfg.C,
                )
                nisa.dma_copy(dst=staged[:, nl.ds(lane_lo, n)], src=src_view, dge_mode=nisa.dge_mode.none)
            nisa.dma_transpose(src=staged, dst=flat[:, nl.ds(grp_idx * slot + s_off, s_len)], axes=(1, 0))
            s_off += s_len

    if slot == S and win[0] == D and win[1] == H and win[2] == W:
        return _as_tile(flat, part, grp, D, H, W)

    cur = _alloc_tile(part, grp, win[0], win[1], win[2], cfg.dtype)
    nisa.memset(cur[...], value=_INIT_VALUE, engine=nisa.engine.gpsimd)
    cur_flat = cur.reshape((part, grp * vol)) if grp > 1 else cur
    for grp_idx in range(grp):
        nisa.tensor_copy(
            dst=cur_flat.ap(
                pattern=[[grp * vol, part], [win[1] * win[2], D], [win[2], H], [1, W]],
                offset=grp_idx * vol + (pad_lo[0] * win[1] + pad_lo[1]) * win[2] + pad_lo[2],
            ),
            src=flat.ap(
                pattern=[[grp * slot, part], [H * W, D], [W, H], [1, W]],
                offset=grp_idx * slot,
            ),
        )
    return cur


def _store_tile(out, cfg, out_str, res, part, grp, plane_off, tile):
    """Scatter the pooled (part, [grp,] do, ho, wo) tile back in the output layout."""
    Do, Ho, Wo = cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2]
    out_lo, out_lens = tile[0], tile[1]
    do, ho, wo = out_lens[0], out_lens[1], out_lens[2]
    s_b, s_c, s_d, s_h, s_w = out_str[0], out_str[1], out_str[2], out_str[3], out_str[4]
    So = _prod(out_lens)
    whole_plane = do == Do and ho == Ho and wo == Wo

    if cfg.out_transpose and whole_plane:
        _store_tile_transposed(out, cfg, res, part, grp, plane_off, So)
        return

    if cfg.group_mode == _GROUP_LANE and grp > 1 and whole_plane:
        nisa.dma_copy(
            dst=out.ap(pattern=[[grp * So, part], [1, grp * So]], offset=plane_off * s_c),
            src=res.reshape((part, grp * So)),
            dge_mode=nisa.dge_mode.none,
        )
        return

    res_flat = res.reshape((part, grp * So)) if grp > 1 else res
    base = out_lo[0] * s_d + out_lo[1] * s_h + out_lo[2] * s_w
    for grp_idx in range(grp):
        runs = _batch_runs(_group_plane(plane_off, part, grp_idx, cfg), part, cfg.C)
        for run_idx in range(len(runs)):
            run = runs[run_idx]
            lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
            dst_view = out.ap(
                pattern=[[s_c, n], [s_d, do], [s_h, ho], [s_w, wo]],
                offset=batch_idx * s_b + chan * s_c + base,
            )
            if grp > 1:
                src_view = res_flat.ap(
                    pattern=[[grp * So, n], [ho * wo, do], [wo, ho], [1, wo]],
                    offset=lane_lo * grp * So + grp_idx * So,
                )
            else:
                src_view = res[nl.ds(lane_lo, n), :, :, :]
            nisa.dma_copy(dst=dst_view, src=src_view, dge_mode=nisa.dge_mode.none)


def _store_tile_transposed(out, cfg, res, part, grp, plane_off, So):
    """Transpose whole pooled planes back to channels-last and store them."""
    flat = res.reshape((part, grp * So))
    for grp_idx in range(grp):
        runs = _batch_runs(_group_plane(plane_off, part, grp_idx, cfg), part, cfg.C)
        s_off = 0
        while s_off < So:
            s_len = min(cfg.partition_limit, So - s_off)
            staged = nl.ndarray((s_len, part), dtype=cfg.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(src=flat[:, nl.ds(grp_idx * So + s_off, s_len)], dst=staged, axes=(1, 0))
            for run_idx in range(len(runs)):
                run = runs[run_idx]
                lane_lo, batch_idx, chan, n = run[0], run[1], run[2], run[3]
                dst_view = out.ap(
                    pattern=[[cfg.C, s_len], [1, n]],
                    offset=batch_idx * So * cfg.C + chan + s_off * cfg.C,
                )
                nisa.dma_copy(dst=dst_view, src=staged[:, nl.ds(lane_lo, n)], dge_mode=nisa.dge_mode.none)
            s_off += s_len


def _shard_range(total, n_prgs, prg_id):
    """Contiguous [start, start + size) slice of total owned by this program."""
    per = div_ceil(total, n_prgs)
    start = min(prg_id * per, total)
    return start, min(per, total - start)


def _work_items(cfg):
    """Return this program's (part, grp, plane_off, tile) work items."""
    if cfg.group_mode == _GROUP_BATCH:
        return _plan_batch_tiles(cfg.B, cfg.C, cfg.batch_group, cfg.partition_limit, cfg.n_prgs, cfg.prg_id)

    planes = cfg.B * cfg.C
    shard_start, shard_planes = _shard_range(planes, cfg.n_prgs, cfg.prg_id)
    tiles = cfg.tiles
    items = []
    for tile_idx in range(len(tiles)):
        plane_off = shard_start
        shard_end = shard_start + shard_planes
        while plane_off < shard_end:
            rem = shard_end - plane_off
            if rem >= cfg.partition_limit:
                part = cfg.partition_limit
                grp = min(cfg.group, rem // cfg.partition_limit)
            else:
                part, grp = rem, 1
            items.append((part, grp, plane_off, tiles[tile_idx]))
            plane_off += part * grp
    return items


def _plan_batch_tiles(B, C, batch_group, partition_limit, n_prgs, prg_id):
    """Work items for the batch-packed mode, whose tile is always the whole plane."""
    tiles = []
    item_idx = 0
    b0 = 0
    while b0 < B:
        grp = min(batch_group, B - b0)
        chan = 0
        while chan < C:
            part = min(partition_limit, C - chan)
            if item_idx % n_prgs == prg_id:
                tiles.append((part, grp, b0 * C + chan, None))
            item_idx += 1
            chan += part
        b0 += grp
    return tiles


def _lane_groups(planes, group, partition_limit, n_prgs, prg_id):
    """Number of lane groups this program's plane shard is split into."""
    shard_start, shard_planes = _shard_range(planes, n_prgs, prg_id)
    count = 0
    off = shard_start
    end = shard_start + shard_planes
    while off < end:
        rem = end - off
        if rem >= partition_limit:
            part = partition_limit
            grp = min(group, rem // partition_limit)
        else:
            part, grp = rem, 1
        count += 1
        off += part * grp
    return count


def _reduce_all_axes(cur, cfg, grp, out_lens, ks_eff, st_eff, scale):
    """Separable sum-reduce over all pooled axes; scale None keeps the fp32 sum."""
    n_lead = 1 if grp > 1 else 0

    n_fold = 0
    for axis_idx in range(_POOL_RANK - 1, -1, -1):
        if out_lens[axis_idx] == 1 and ks_eff[axis_idx] == cur.shape[1 + n_lead + axis_idx]:
            n_fold += 1
        else:
            break

    for axis_idx in range(_POOL_RANK - n_fold - 1, -1, -1):
        if ks_eff[axis_idx] == 1 and out_lens[axis_idx] == cur.shape[1 + n_lead + axis_idx]:
            continue
        cur = _sum_reduce_axis(cur, n_lead + axis_idx, ks_eff[axis_idx], st_eff[axis_idx], out_lens[axis_idx])
    if n_fold > 0:
        cur = _sum_reduce_tail(cur, n_lead + _POOL_RANK - n_fold)

    if scale == None:
        return cur

    part = cur.shape[0]
    n_free = _prod(cur.shape[1:])
    res = _alloc_tile(part, grp, out_lens[0], out_lens[1], out_lens[2], cfg.dtype)
    flat_pattern = [[n_free, part], [1, n_free]]
    nisa.tensor_scalar(
        dst=res.ap(pattern=flat_pattern),
        data=cur.ap(pattern=flat_pattern),
        op0=nl.multiply,
        operand0=scale,
        engine=nisa.engine.scalar,
    )
    return res


def _sum_reduce_tail(cur, first):
    """Fold free dims first each pooling to a single output into one reduce."""
    part = cur.shape[0]
    dims = list(cur.shape[1:])
    num_out = _prod(dims[:first])
    kernel_len = _prod(dims[first:])

    out_dims = list(dims)
    for idx in range(first, len(out_dims)):
        out_dims[idx] = 1
    out = _alloc_reduce_out(part, out_dims)

    src_pattern = [[num_out * kernel_len, part]]
    dst_pattern = [[num_out, part]]
    if num_out > 1:
        src_pattern.append([kernel_len, num_out])
        dst_pattern.append([1, num_out])
    src_pattern.append([1, kernel_len])
    if len(dst_pattern) == 1:
        dst_pattern.append([1, 1])
    nisa.tensor_reduce(
        dst=out.ap(pattern=dst_pattern),
        op=nl.add,
        data=cur.ap(pattern=src_pattern),
        axis=len(src_pattern) - 1,
    )
    return out


def _alloc_reduce_out(part, out_dims):
    """Allocate an fp32 reduce destination of shape (part, *out_dims) (3 or 4 free dims)."""
    if len(out_dims) == _POOL_RANK:
        return nl.ndarray((part, out_dims[0], out_dims[1], out_dims[2]), dtype=nl.float32, buffer=nl.sbuf)
    return nl.ndarray((part, out_dims[0], out_dims[1], out_dims[2], out_dims[3]), dtype=nl.float32, buffer=nl.sbuf)


def _sum_reduce_axis(cur, axis_idx, kernel_len, stride_len, out_len):
    """Sum-reduce free-dim axis axis_idx of SBUF tile cur with a length-kernel_len window."""
    part = cur.shape[0]
    dims = list(cur.shape[1:])
    num_in = _prod(dims[axis_idx + 1 :])
    num_out = _prod(dims[:axis_idx])
    axis_len = dims[axis_idx]
    f_in = num_out * axis_len * num_in

    out_dims = list(dims)
    out_dims[axis_idx] = out_len
    f_out = num_out * out_len * num_in
    out = _alloc_reduce_out(part, out_dims)

    win_step = num_in
    pos_step = stride_len * num_in

    if num_out > 1 and num_in > 1 and out_len > 1:
        for out_idx in range(num_out):
            data = cur.ap(
                pattern=[[f_in, part], [pos_step, out_len], [1, num_in], [win_step, kernel_len]],
                offset=out_idx * axis_len * num_in,
            )
            dst = out.ap(
                pattern=[[f_out, part], [num_in, out_len], [1, num_in]],
                offset=out_idx * out_len * num_in,
            )
            nisa.tensor_reduce(dst=dst, op=nl.add, data=data, axis=3)
        return out

    src_pattern = [[f_in, part]]
    dst_pattern = [[f_out, part]]
    if num_out > 1:
        src_pattern.append([axis_len * num_in, num_out])
        dst_pattern.append([out_len * num_in, num_out])
    if out_len > 1:
        src_pattern.append([pos_step, out_len])
        dst_pattern.append([num_in, out_len])
    if num_in > 1:
        src_pattern.append([1, num_in])
        dst_pattern.append([1, num_in])
    src_pattern.append([win_step, kernel_len])
    if len(dst_pattern) == 1:
        dst_pattern.append([1, 1])
    nisa.tensor_reduce(
        dst=out.ap(pattern=dst_pattern),
        op=nl.add,
        data=cur.ap(pattern=src_pattern),
        axis=len(src_pattern) - 1,
    )
    return out


def _prod(values):
    """Return the product of all elements in an iterable."""
    product = 1
    for value in values:
        product *= value
    return product


def _per_axis(value, rank):
    """A scalar broadcasts to every spatial axis."""
    return [value] * rank if isinstance(value, int) else list(value)


def _out_dim(in_len, pad, kernel_len, stride_len):
    """Pooled output length of one axis."""
    return (in_len + 2 * pad - kernel_len) // stride_len + 1


def _tile_bytes(ks, st, out_lens, grp, dtype_bytes, in_transpose, split):
    """Peak SBUF bytes per lane held by one work item on the tile out_lens."""
    win = _win_lens(out_lens, ks, st)
    win_vol = win[0] * win[1] * win[2]
    out_vol = out_lens[0] * out_lens[1] * out_lens[2]

    total = dtype_bytes * grp * win_vol
    if in_transpose:
        total += dtype_bytes * grp * win_vol
    total += _DTYPE_BYTES_4 * grp * (win[0] * win[1] * out_lens[2] + win[0] * out_lens[1] * out_lens[2] + out_vol)
    total += dtype_bytes * grp * out_vol
    if split:
        total += _DTYPE_BYTES_4 * grp * out_vol
    return total


def _tile_ap_ok(ks, st, out_lens, grp):
    """Whether every access-pattern element count of this tile fits a uint16."""
    win = _win_lens(out_lens, ks, st)
    win_vol = grp * win[0] * win[1] * win[2]
    out_vol = grp * out_lens[0] * out_lens[1] * out_lens[2]
    return win_vol <= _AP_NUM_MAX and out_vol <= _AP_NUM_MAX


def _plan_window_split(ks, st, in_spatial, dtype_bytes, in_transpose, budget):
    """(axis, piece) splitting one window into summable pieces that fit budget."""
    single = (1, 1, 1)
    for axis in range(_POOL_RANK):
        if ks[axis] <= 1:
            continue
        lo, hi = 1, min(ks[axis], in_spatial[axis])
        best = 0
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            kernel = []
            for pool_axis in range(_POOL_RANK):
                kernel.append(mid if pool_axis == axis else ks[pool_axis])
            if (
                _tile_ap_ok(tuple(kernel), st, single, 1)
                and _tile_bytes(tuple(kernel), st, single, 1, dtype_bytes, in_transpose, True) <= budget
            ):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best > 0:
            return axis, best
    return 0, 0


@dataclass(frozen=True)
class AvgPool3DConfig(nl.NKIObject):
    """Configuration for the 3D average-pool kernel."""

    # Dimensions
    B: int
    C: int
    D: int
    H: int
    W: int
    ks: tuple
    st: tuple
    pd: tuple
    no_pad: tuple
    out_spatial: tuple
    dtype: type
    data_format: str
    output_format: str
    scale: float

    # Tiling
    partition_limit: int
    tiles: list
    group: int
    batch_chunks: int
    transpose_align_elems: int
    two_byte: bool
    group_mode: str
    batch_group: int
    split_axis: int
    split_piece: int

    # Transpose
    in_transpose: bool
    out_transpose: bool

    # Tuning
    copy_engine: type

    # Sharding
    n_prgs: int
    prg_id: int


def _build_config(src_tensor, kernel_size, stride, padding, data_format, output_format):
    """Parse inputs, derive tile geometry, and log the config."""
    if stride == None:
        stride = kernel_size
    if output_format == None:
        output_format = data_format

    max_group_planes = 512  # Cap on planes packed onto a tile's free axis
    free_elem_budget = 16384  # Target free-axis elements per lane for a group-packed tile
    pipeline_tiles = 4  # Tiles per shard, so tile k's reduces overlap tile k+1's DMA
    batch_chunks = 2  # Chunks per batch-packed tile, so load i+1 overlaps reduce i
    batch_group_free_elems = 4096  # Free-axis budget (batch_group * spatial vol) for batch packing

    partition_limit = nl.tile_size.pmax
    total_sbuf = nl.tile_size.total_available_sbuf_size

    _, n_prgs, prg_id = get_program_sharding_info()

    ks = tuple(_per_axis(kernel_size, _POOL_RANK))
    st = tuple(_per_axis(stride, _POOL_RANK))
    pd = tuple(_per_axis(padding, _POOL_RANK))

    shape = list(src_tensor.shape)
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

    two_byte = str(src_tensor.dtype) == str(nl.bfloat16) or str(src_tensor.dtype) == str(nl.float16)
    dtype_bytes = _DTYPE_BYTES_2 if two_byte else _DTYPE_BYTES_4
    in_transpose = two_byte and data_format == "NDHWC"
    out_transpose = two_byte and output_format == "NDHWC"
    channels_first = data_format == "NCDHW" and output_format == "NCDHW"

    def fits(out_lens):
        return (
            _tile_ap_ok(ks, st, out_lens, 1)
            and _tile_bytes(ks, st, out_lens, 1, dtype_bytes, in_transpose, False) <= total_sbuf
        )

    chunk_axis, chunk = _plan_chunk(out_spatial, fits)
    budget_lens = _tile_lens(out_spatial, chunk_axis, chunk)

    split_axis, split_piece = 0, 0
    if not fits(budget_lens):
        budget_lens = (1, 1, 1)
        split_axis, split_piece = _plan_window_split(ks, st, in_spatial, dtype_bytes, in_transpose, total_sbuf)

    planes = B * C
    lens = [budget_lens[0], budget_lens[1], budget_lens[2]]
    if split_piece == 0 and planes < partition_limit:
        occupancy = planes
        for axis_idx in range(_POOL_RANK - 1):
            if occupancy >= partition_limit or out_spatial[axis_idx] <= 1:
                continue
            chunks = min(out_spatial[axis_idx], div_ceil(partition_limit, occupancy))
            lens[axis_idx] = min(lens[axis_idx], div_ceil(out_spatial[axis_idx], chunks))
            occupancy *= chunks
    lens = (lens[0], lens[1], lens[2])

    tiles = _spatial_tiles(out_spatial, in_spatial, ks, st, pd, lens)
    tiled = _is_tiled(out_spatial, lens)
    item_bytes = _tile_bytes(ks, st, lens, 1, dtype_bytes, in_transpose, split_piece > 0)

    group = 1
    if not tiled and split_piece == 0:
        _, shard_planes = _shard_range(planes, n_prgs, prg_id)
        win = _win_lens(lens, ks, st)
        group = max(1, min(max_group_planes, free_elem_budget // max(1, win[0] * win[1] * win[2])))
        if group > 1:
            group = max(1, min(group, div_ceil(shard_planes, pipeline_tiles) // partition_limit))
        while group > 1 and not (
            _tile_ap_ok(ks, st, lens, group)
            and _tile_bytes(ks, st, lens, group, dtype_bytes, in_transpose, False) <= total_sbuf
        ):
            group = group // 2

    win_full = _win_lens(lens, ks, st)
    lane_fast = win_full[0] == D and win_full[1] == H and win_full[2] == W
    lane_ok = channels_first and group > 1 and C % group == 0 and lane_fast and not tiled

    no_pad_all = pd[0] == 0 and pd[1] == 0 and pd[2] == 0
    spatial_vol = D * H * W
    full_window = ks[0] == D and ks[1] == H and ks[2] == W
    single_out = out_spatial[0] == 1 and out_spatial[1] == 1 and out_spatial[2] == 1
    batch_eligible = no_pad_all and full_window and single_out and spatial_vol > 0 and B > 1
    batch_group = _batch_group_factor(B, spatial_vol, batch_group_free_elems, max_group_planes) if batch_eligible else 1

    if batch_group > 1 and not lane_ok:
        group_mode = _GROUP_BATCH
        group = 1
        split_piece = 0
        tiles = _spatial_tiles(out_spatial, in_spatial, ks, st, pd, out_spatial)
    else:
        batch_group = 1
        group_mode = _GROUP_LANE if lane_ok else _GROUP_BLOCK

    scale = 1.0 / float(_prod(ks))

    cfg = AvgPool3DConfig(
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        ks=ks,
        st=st,
        pd=pd,
        no_pad=(0, 0, 0),
        out_spatial=out_spatial,
        dtype=src_tensor.dtype,
        data_format=data_format,
        output_format=output_format,
        scale=scale,
        partition_limit=partition_limit,
        tiles=tiles,
        group=group,
        batch_chunks=batch_chunks,
        transpose_align_elems=_TRANSPOSE_ALIGN_BYTES // dtype_bytes,
        two_byte=two_byte,
        group_mode=group_mode,
        batch_group=batch_group,
        split_axis=split_axis,
        split_piece=split_piece,
        in_transpose=in_transpose,
        out_transpose=out_transpose,
        copy_engine=nisa.engine.scalar,
        n_prgs=n_prgs,
        prg_id=prg_id,
    )

    if group_mode == _GROUP_BATCH:
        local_items = len(_plan_batch_tiles(B, C, batch_group, partition_limit, n_prgs, prg_id))
    else:
        local_items = len(tiles) * _lane_groups(planes, group, partition_limit, n_prgs, prg_id)

    logger = get_logger("avg_pooling_3d")
    logger.info(
        "AvgPool3DConfig: "
        "data_format=" + str(data_format) + ", "
        "output_format=" + str(output_format) + ", "
        "B=" + str(B) + ", C=" + str(C) + ", D=" + str(D) + ", H=" + str(H) + ", W=" + str(W) + ", "
        "kernel_size=" + str(ks) + ", stride=" + str(st) + ", padding=" + str(pd) + ", "
        "out_spatial=" + str(out_spatial) + ", dtype=" + str(src_tensor.dtype) + ", "
        "partition_limit=" + str(partition_limit) + ", "
        "tile_lens=" + str(lens) + ", spatial_tiles=" + str(len(tiles)) + ", "
        "tile_bytes=" + str(item_bytes) + "/" + str(total_sbuf) + ", "
        "split_axis=" + str(split_axis) + ", split_piece=" + str(split_piece) + ", "
        "in_transpose=" + str(in_transpose) + ", "
        "out_transpose=" + str(out_transpose) + ", "
        "group=" + str(group) + ", "
        "group_mode=" + str(group_mode) + ", "
        "batch_group=" + str(batch_group) + ", "
        "lnc(n_prgs)=" + str(n_prgs) + ", "
        "shard_id(prg_id)=" + str(prg_id) + ", "
        "local_work_items=" + str(local_items)
    )

    return cfg


def _batch_group_factor(B, spatial_vol, free_elem_budget, max_group_planes):
    """Batches packed onto the free axis: bounded by the free budget and by B."""
    max_grp = max(1, min(max_group_planes, free_elem_budget // max(1, spatial_vol)))
    grp = min(B, max_grp)
    while grp > 1 and B % grp != 0:
        grp -= 1
    return max(1, grp)
