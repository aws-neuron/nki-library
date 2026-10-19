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

"""3D max pooling for NeuronCore using tensor_reduce on the Vector Engine."""

from dataclasses import dataclass
from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from ...core.utils.logging import get_logger

_POOL_RANK = 3  # Number of pooled spatial axes (D, H, W)
_AP_NUM_MAX = 65535  # A hardware access-pattern element count is a uint16
_DTYPE_BYTES_2 = 2  # Byte width of a 2-byte element dtype (bf16/fp16)
_DTYPE_BYTES_4 = 4  # Byte width of a 4-byte element dtype (fp32)
_TRANSPOSE_ALIGN = 32  # Byte alignment required by dma_transpose buffers
_INTERLEAVE_SECTIONS = 2  # Sections used when a tile is small enough to double-buffer


@nki.jit
def max_pooling_3d(
    src_tensor: nl.NkiTensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    data_format: str = "NCDHW",
    output_format: Optional[str] = None,
) -> nl.NkiTensor:
    """3D max pool kernel over the (D, H, W) axes. Pooling runs independently per (batch, channel).

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
        padding (int or tuple): Implicit -inf padding (pD, pH, pW).
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
            res = reduce_all_axes(cur, tile)
            store_window(out, res, lane group, tile)
        return out
    """
    cfg = _build_config(src_tensor, kernel_size, stride, padding, data_format, output_format)

    B, C = cfg.B, cfg.C
    Do, Ho, Wo = cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2]
    dtype = cfg.dtype
    So = Do * Ho * Wo

    logger = get_logger("max_pooling_3d")
    sbm = SbufManager(0, cfg.total_sbuf, logger=logger)

    if cfg.output_format == "NDHWC":
        out = nl.ndarray((B, Do, Ho, Wo, C), dtype=dtype, buffer=nl.shared_hbm)
    else:
        out = nl.ndarray((B, C, Do, Ho, Wo), dtype=dtype, buffer=nl.shared_hbm)

    sbm.open_scope(interleave_degree=cfg.tile_interleave, name="pool_tile")

    items = _work_items(cfg)
    for item_idx in range(len(items)):
        item = items[item_idx]
        part = item[0]
        grp = item[1]
        batch_idx = item[2]
        c_off = item[3]
        c_tile = item[4]
        n_batch = item[5]
        tile = item[6]
        n_lead = 1 if grp > 1 else 0

        sbm.open_scope(interleave_degree=cfg.load_interleave, name="pool_load")

        # Step 1: load this tile's input window from HBM
        cur = _load_window(src_tensor, cfg, sbm, part, grp, batch_idx, c_off, c_tile, n_batch, tile)

        # Step 2: max reduction over all pooled axes
        res = _reduce_all_axes(cur, n_lead, cfg, sbm, tile)

        # Step 3: store the tile's output window to HBM
        _store_window(out, cfg, sbm, res, part, grp, batch_idx, c_off, c_tile, n_batch, tile, So)

        sbm.close_scope()  # load
        sbm.increment_section()

    sbm.close_scope()  # tile

    return out


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


def _nominal_in_lens(out_lens, in_spatial, ks, st):
    """Largest per-axis input extent a tile with these output extents can need."""
    lens = []
    for pool_axis in range(_POOL_RANK):
        span = (out_lens[pool_axis] - 1) * st[pool_axis] + ks[pool_axis]
        lens.append(min(in_spatial[pool_axis], span))
    return tuple(lens)


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


def _load_window(src_tensor, cfg, sbm, part, grp, batch_idx, c_off, c_tile, n_batch, tile):
    """Load one lane group's window for tile into a (part, [grp,] di, hi, wi) SBUF tile."""
    C, D, H, W = cfg.C, cfg.D, cfg.H, cfg.W
    S = D * H * W
    in_batch = S * C
    in_lo, in_lens = tile[2], tile[3]
    in_vol = _prod(in_lens)
    di, hi, wi = in_lens[0], in_lens[1], in_lens[2]
    in_base = _flat_offset(in_lo, (D, H, W))
    dtype = cfg.dtype

    if cfg.packed:
        if grp > 1:
            src_view = src_tensor.ap(
                pattern=[[grp * S, part], [S, grp], [H * W, di], [W, hi], [1, wi]],
                offset=c_off * S + in_base,
            )
            cur = sbm.alloc_stack((part, grp, di, hi, wi), dtype=dtype, buffer=nl.sbuf)
        else:
            src_view = src_tensor.ap(
                pattern=[[S, part], [H * W, di], [W, hi], [1, wi]],
                offset=c_off * S + in_base,
            )
            cur = sbm.alloc_stack((part, di, hi, wi), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cur[...], src=src_view, dge_mode=nisa.dge_mode.none)
        return cur

    runs = _box_runs(in_lo, in_lens, (D, H, W))
    if cfg.data_format == "NDHWC" and _is_2byte(dtype) and len(runs) == 1:
        transposed = sbm.alloc_stack((part, in_vol), dtype=dtype, buffer=nl.sbuf, align=_TRANSPOSE_ALIGN)
        for batch_off in range(n_batch):
            src_view = src_tensor.ap(
                pattern=[[C, in_vol], [1, c_tile]],
                offset=(batch_idx + batch_off) * in_batch + in_base * C + c_off,
            )
            nisa.dma_transpose(src=src_view, dst=transposed[nl.ds(batch_off * c_tile, c_tile), :], axes=(1, 0))
        return transposed.reshape((part, di, hi, wi))

    cur = sbm.alloc_stack((part, di, hi, wi), dtype=dtype, buffer=nl.sbuf)
    for batch_off in range(n_batch):
        if cfg.data_format == "NDHWC":
            in_sD, in_sH, in_sW = H * W * C, W * C, C
            src_view = src_tensor.ap(
                pattern=[[1, c_tile], [in_sD, di], [in_sH, hi], [in_sW, wi]],
                offset=(batch_idx + batch_off) * in_batch
                + c_off
                + in_lo[0] * in_sD
                + in_lo[1] * in_sH
                + in_lo[2] * in_sW,
            )
        else:
            src_view = src_tensor.ap(
                pattern=[[S, c_tile], [H * W, di], [W, hi], [1, wi]],
                offset=(batch_idx + batch_off) * in_batch + c_off * S + in_base,
            )
        nisa.dma_copy(
            dst=cur[nl.ds(batch_off * c_tile, c_tile), :, :, :],
            src=src_view,
            dge_mode=nisa.dge_mode.none,
        )
    return cur


def _store_window(out, cfg, sbm, res, part, grp, batch_idx, c_off, c_tile, n_batch, tile, So):
    """Scatter the pooled (part, [grp,] do, ho, wo) tile back in the output layout."""
    C = cfg.C
    Do, Ho, Wo = cfg.out_spatial[0], cfg.out_spatial[1], cfg.out_spatial[2]
    out_lo, out_lens = tile[0], tile[1]
    do, ho, wo = out_lens[0], out_lens[1], out_lens[2]
    out_base = _flat_offset(out_lo, (Do, Ho, Wo))
    out_vol = _prod(out_lens)
    P_MAX = cfg.partition_limit

    if cfg.packed:
        if grp > 1:
            dst_view = out.ap(
                pattern=[[grp * So, part], [So, grp], [Ho * Wo, do], [Wo, ho], [1, wo]],
                offset=c_off * So + out_base,
            )
        else:
            dst_view = out.ap(
                pattern=[[So, part], [Ho * Wo, do], [Wo, ho], [1, wo]],
                offset=c_off * So + out_base,
            )
        nisa.dma_copy(dst=dst_view, src=res, dge_mode=nisa.dge_mode.none)
        return

    out_batch = So * C if cfg.output_format == "NDHWC" else C * So
    runs = _box_runs(out_lo, out_lens, (Do, Ho, Wo))

    if cfg.output_format == "NDHWC" and _is_2byte(cfg.dtype) and len(runs) == 1:
        _store_ndhwc_transposed(out, cfg, sbm, res, part, batch_idx, c_off, c_tile, n_batch, out_base, out_vol, P_MAX)
        return

    if cfg.output_format == "NDHWC":
        out_sD, out_sH, out_sW = Ho * Wo * C, Wo * C, C
        for batch_off in range(n_batch):
            dst_view = out.ap(
                pattern=[[1, c_tile], [out_sD, do], [out_sH, ho], [out_sW, wo]],
                offset=(batch_idx + batch_off) * out_batch
                + c_off
                + out_lo[0] * out_sD
                + out_lo[1] * out_sH
                + out_lo[2] * out_sW,
            )
            nisa.dma_copy(
                dst=dst_view,
                src=res[nl.ds(batch_off * c_tile, c_tile), :, :, :],
                dge_mode=nisa.dge_mode.none,
            )
        return

    for batch_off in range(n_batch):
        dst_view = out.ap(
            pattern=[[So, c_tile], [Ho * Wo, do], [Wo, ho], [1, wo]],
            offset=(batch_idx + batch_off) * out_batch + c_off * So + out_base,
        )
        nisa.dma_copy(
            dst=dst_view,
            src=res[nl.ds(batch_off * c_tile, c_tile), :, :, :],
            dge_mode=nisa.dge_mode.none,
        )


def _store_ndhwc_transposed(out, cfg, sbm, res, part, batch_idx, c_off, c_tile, n_batch, out_base, out_vol, P_MAX):
    """PE-transpose the tile back to channels-last and store it in (spatial, C) order."""
    C = cfg.C
    out_batch = cfg.out_spatial[0] * cfg.out_spatial[1] * cfg.out_spatial[2] * C
    res_flat = res.reshape((part, out_vol))
    n_full, rem = divmod(out_vol, P_MAX)

    if n_full > 0:
        big = sbm.alloc_stack((P_MAX, n_full, part), dtype=cfg.dtype, buffer=nl.sbuf, align=_TRANSPOSE_ALIGN)
        evict_off = 0
        while evict_off < n_full:
            evict_len = min(cfg.evict_group, n_full - evict_off)
            ps = nl.ndarray((P_MAX, evict_len, part), dtype=cfg.dtype, buffer=nl.psum)
            for evict_idx in range(evict_len):
                nisa.nc_transpose(
                    dst=ps[:, evict_idx, :],
                    data=res_flat[:, nl.ds((evict_off + evict_idx) * P_MAX, P_MAX)],
                )
            nisa.tensor_copy(dst=big[:, nl.ds(evict_off, evict_len), :], src=ps, engine=nisa.engine.scalar)
            evict_off += evict_len
        for batch_off in range(n_batch):
            dst_view = out.ap(
                pattern=[[C, P_MAX], [P_MAX * C, n_full], [1, c_tile]],
                offset=(batch_idx + batch_off) * out_batch + out_base * C + c_off,
            )
            nisa.dma_copy(
                dst=dst_view,
                src=big[:, :, nl.ds(batch_off * c_tile, c_tile)],
                dge_mode=nisa.dge_mode.none,
            )
    if rem > 0:
        remb = sbm.alloc_stack((rem, part), dtype=cfg.dtype, buffer=nl.sbuf, align=_TRANSPOSE_ALIGN)
        psr = nl.ndarray((rem, part), dtype=cfg.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psr, data=res_flat[:, nl.ds(n_full * P_MAX, rem)])
        nisa.tensor_copy(dst=remb, src=psr, engine=nisa.engine.scalar)
        for batch_off in range(n_batch):
            dst_view = out.ap(
                pattern=[[C, rem], [1, c_tile]],
                offset=(batch_idx + batch_off) * out_batch + (out_base + n_full * P_MAX) * C + c_off,
            )
            nisa.dma_copy(
                dst=dst_view,
                src=remb[:, nl.ds(batch_off * c_tile, c_tile)],
                dge_mode=nisa.dge_mode.none,
            )


def _work_items(cfg):
    """Return (part, grp, batch_idx, c_off, c_tile, n_batch, tile) work items for this shard."""
    P_MAX = cfg.partition_limit
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    tiles = cfg.tiles
    work_item_idx = 0
    items = []
    if cfg.packed:
        planes = cfg.B * cfg.C
        off = 0
        while off < planes:
            rem = planes - off
            if rem >= P_MAX:
                grp = min(cfg.group, rem // P_MAX)
                part = P_MAX
            else:
                grp, part = 1, rem
            for tile_idx in range(len(tiles)):
                if work_item_idx % n_prgs == prg_id:
                    items.append((part, grp, 0, off, part, 1, tiles[tile_idx]))
                work_item_idx += 1
            off += part * grp
    elif cfg.batch_pack > 1:
        batch_idx = 0
        while batch_idx < cfg.B:
            n_batch = min(cfg.batch_pack, cfg.B - batch_idx)
            for tile_idx in range(len(tiles)):
                if work_item_idx % n_prgs == prg_id:
                    items.append((n_batch * cfg.C, 1, batch_idx, 0, cfg.C, n_batch, tiles[tile_idx]))
                work_item_idx += 1
            batch_idx += n_batch
    else:
        for batch_idx in range(cfg.B):
            c_off = 0
            while c_off < cfg.C:
                c_tile = min(P_MAX, cfg.C - c_off)
                for tile_idx in range(len(tiles)):
                    if work_item_idx % n_prgs == prg_id:
                        items.append((c_tile, 1, batch_idx, c_off, c_tile, 1, tiles[tile_idx]))
                    work_item_idx += 1
                c_off += c_tile
    return items


def _reduce_all_axes(cur, n_lead, cfg, sbm, tile):
    """Separable max-reduce of a loaded window down to the tile's output extents."""
    part = cur.shape[0]
    lead = tuple(cur.shape[1 : 1 + n_lead])
    out_lens, pad_lo = tile[1], tile[4]
    result = sbm.alloc_stack((part,) + lead + out_lens, dtype=cfg.dtype, buffer=nl.sbuf)

    sbm.open_scope(interleave_degree=cfg.reduce_interleave, name="pool_reduce")
    for axis_idx in range(cfg.pool_rank):
        if _is_identity_axis(cfg.ks[axis_idx], cfg.st[axis_idx], pad_lo[axis_idx]):
            continue
        cur = _max_reduce_axis(
            cur,
            n_lead + axis_idx,
            cfg.ks[axis_idx],
            cfg.st[axis_idx],
            pad_lo[axis_idx],
            out_lens[axis_idx],
            cfg.dtype,
            sbm,
        )
        sbm.increment_section()
    nisa.tensor_copy(dst=result[...], src=cur[...], engine=nisa.engine.scalar)
    sbm.close_scope()
    return result


def _is_identity_axis(kernel_len, stride_len, pad_len):
    """Whether an axis passes through unchanged, so its reduce can be skipped."""
    return kernel_len == 1 and stride_len == 1 and pad_len == 0


def _max_reduce_axis(cur, axis_idx, kernel_len, stride_len, pad_len, out_len, out_dtype, sbm):
    """Max-reduce free-dim axis axis_idx of SBUF tile cur with a length-kernel_len window"""
    part = cur.shape[0]
    dims = list(cur.shape[1:])
    num_in = _prod(dims[axis_idx + 1 :])
    num_out = _prod(dims[:axis_idx])
    axis_len = dims[axis_idx]
    f_in = num_out * axis_len * num_in

    out_dims = list(dims)
    out_dims[axis_idx] = out_len
    f_out = num_out * out_len * num_in
    if len(out_dims) == _POOL_RANK:
        out = sbm.alloc_stack((part, out_dims[0], out_dims[1], out_dims[2]), dtype=out_dtype, buffer=nl.sbuf)
    else:
        out = sbm.alloc_stack(
            (part, out_dims[0], out_dims[1], out_dims[2], out_dims[3]), dtype=out_dtype, buffer=nl.sbuf
        )

    win_step = num_in
    pos_step = stride_len * num_in

    n_left = div_ceil(pad_len, stride_len)
    o_hi = (axis_len - kernel_len + pad_len) // stride_len
    if o_hi > out_len - 1:
        o_hi = out_len - 1
    interior_cnt = o_hi - n_left + 1

    if num_out > 1 and num_in > 1:
        for out_idx in range(num_out):
            in_blk = out_idx * axis_len * num_in
            out_blk = out_idx * out_len * num_in
            if interior_cnt > 0:
                a0 = n_left * stride_len - pad_len
                dst = out.ap(
                    pattern=[[f_out, part], [num_in, interior_cnt], [1, num_in]],
                    offset=out_blk + n_left * num_in,
                )
                srcs = [
                    cur.ap(
                        pattern=[[f_in, part], [pos_step, interior_cnt], [1, num_in]],
                        offset=in_blk + a0 * num_in + window_idx * win_step,
                    )
                    for window_idx in range(kernel_len)
                ]
                _max_tree(dst, srcs)
            for out_pos_idx in _boundary_positions(n_left, o_hi, out_len):
                s0 = max(0, out_pos_idx * stride_len - pad_len)
                width = min(axis_len, out_pos_idx * stride_len - pad_len + kernel_len) - s0
                dst = out.ap(
                    pattern=[[f_out, part], [num_in, 1], [1, num_in]],
                    offset=out_blk + out_pos_idx * num_in,
                )
                srcs = [
                    cur.ap(
                        pattern=[[f_in, part], [pos_step, 1], [1, num_in]],
                        offset=in_blk + s0 * num_in + window_idx * win_step,
                    )
                    for window_idx in range(width)
                ]
                _max_tree(dst, srcs)
        return out

    if interior_cnt > 0:
        a0 = n_left * stride_len - pad_len
        _reduce_region(
            cur,
            out,
            part,
            num_out,
            num_in,
            axis_len,
            out_len,
            f_in,
            f_out,
            a0,
            n_left,
            interior_cnt,
            kernel_len,
            win_step,
            pos_step,
        )
    for out_pos_idx in _boundary_positions(n_left, o_hi, out_len):
        s0 = max(0, out_pos_idx * stride_len - pad_len)
        width = min(axis_len, out_pos_idx * stride_len - pad_len + kernel_len) - s0
        _reduce_region(
            cur,
            out,
            part,
            num_out,
            num_in,
            axis_len,
            out_len,
            f_in,
            f_out,
            s0,
            out_pos_idx,
            1,
            width,
            win_step,
            pos_step,
        )
    return out


def _boundary_positions(n_left, o_hi, out_len):
    """Output positions whose window hangs off an edge (needs clamped sub-window)."""
    left = list(range(0, n_left))
    right = list(range(max(o_hi + 1, n_left), out_len))
    return left + right


def _reduce_region(
    cur, out, part, num_out, num_in, axis_len, out_len, f_in, f_out, a0, out_pos, cnt, width, win_step, pos_step
):
    """Max-reduce cnt output positions starting at out_pos from input axis
    start a0 with window width, for the single-nesting (num_out xor num_in) case."""
    src_pattern = [[f_in, part]]
    dst_pattern = [[f_out, part]]
    if num_out > 1:
        src_pattern.append([axis_len * num_in, num_out])
        dst_pattern.append([out_len * num_in, num_out])
    src_pattern.append([pos_step, cnt])
    dst_pattern.append([num_in, cnt])
    if num_in > 1:
        src_pattern.append([1, num_in])
        dst_pattern.append([1, num_in])
    srcs = [cur.ap(pattern=src_pattern, offset=a0 * num_in + window_idx * win_step) for window_idx in range(width)]
    _max_tree(out.ap(pattern=dst_pattern, offset=out_pos * num_in), srcs)


def _max_tree(dst, srcs):
    """Write max over srcs (strided views of equal shape) into dst
    using a (len-1)-op nisa.tensor_tensor(maximum) tree instead of tensor_reduce."""
    if len(srcs) == 1:
        nisa.tensor_copy(dst=dst, src=srcs[0])
        return
    nisa.tensor_tensor(dst=dst, data1=srcs[0], data2=srcs[1], op=nl.maximum)
    for src in srcs[2:]:
        nisa.tensor_tensor(dst=dst, data1=dst, data2=src, op=nl.maximum)


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


def _is_2byte(dtype):
    """dma_transpose requires a 2-byte element dtype."""
    return str(dtype) == str(nl.bfloat16) or str(dtype) == str(nl.float16)


def _tile_bytes(ks, st, pd, in_spatial, out_lens, grp, dtype_bytes, stage_out, partition_limit):
    """Peak SBUF bytes per lane held by one work item on the tile out_lens."""
    in_lens = _nominal_in_lens(out_lens, in_spatial, ks, st)
    in_vol = in_lens[0] * in_lens[1] * in_lens[2]
    out_vol = out_lens[0] * out_lens[1] * out_lens[2]

    dims = [in_lens[0], in_lens[1], in_lens[2]]
    first, second = 0, 0
    for axis_idx in range(_POOL_RANK):
        if _is_identity_axis(ks[axis_idx], st[axis_idx], pd[axis_idx]):
            continue
        dims[axis_idx] = out_lens[axis_idx]
        vol = dims[0] * dims[1] * dims[2]
        if vol > first:
            second = first
            first = vol
        elif vol > second:
            second = vol

    total = grp * (in_vol + out_vol + first + second)
    if stage_out:
        total += grp * out_vol + partition_limit
    return total * dtype_bytes


def _plane_group_count(packed, B, C, group, partition_limit, batch_pack):
    """Number of lane groups the planes are split into, before spatial tiling."""
    if packed:
        planes = B * C
        count = 0
        off = 0
        while off < planes:
            rem = planes - off
            if rem >= partition_limit:
                grp = min(group, rem // partition_limit)
                part = partition_limit
            else:
                grp, part = 1, rem
            count += 1
            off += part * grp
        return count
    if batch_pack > 1:
        return div_ceil(B, batch_pack)
    return B * div_ceil(C, partition_limit)


def _tile_ap_ok(ks, st, in_spatial, out_lens, grp):
    """Whether every access-pattern element count of this tile fits a uint16."""
    in_lens = _nominal_in_lens(out_lens, in_spatial, ks, st)
    in_vol = grp * in_lens[0] * in_lens[1] * in_lens[2]
    out_vol = grp * out_lens[0] * out_lens[1] * out_lens[2]
    return in_vol <= _AP_NUM_MAX and out_vol <= _AP_NUM_MAX


@dataclass(frozen=True)
class MaxPool3DConfig(nl.NKIObject):
    """Configuration for the 3D max-pool kernel."""

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
    packed: bool

    # Tiling
    partition_limit: int
    pool_rank: int
    group: int
    evict_group: int
    tiles: list

    # Interleave degrees
    tile_interleave: int
    load_interleave: int
    reduce_interleave: int

    # SBUF budget
    total_sbuf: int

    # Sharding
    n_prgs: int
    prg_id: int

    # Batches stacked on the partition axis when C < partition_limit
    batch_pack: int = 1


def _build_config(src_tensor, kernel_size, stride, padding, data_format, output_format):
    """Parse inputs, derive tile geometry + interleave degrees, and log the config."""
    if stride == None:
        stride = kernel_size
    if output_format == None:
        output_format = data_format

    free_elem_budget = 16384
    max_group_planes = 512
    pipeline_tiles = 4
    evict_group = 8

    partition_limit = nl.tile_size.pmax
    pool_rank = _POOL_RANK

    _, n_prgs, prg_id = get_program_sharding_info()

    ks = tuple(_per_axis(kernel_size, pool_rank))
    st = tuple(_per_axis(stride, pool_rank))
    pd = tuple(_per_axis(padding, pool_rank))

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

    total_sbuf = nl.tile_size.total_available_sbuf_size
    dtype_bytes = _DTYPE_BYTES_2 if _is_2byte(src_tensor.dtype) else _DTYPE_BYTES_4
    packed = data_format == "NCDHW" and output_format == "NCDHW"
    stage_out = output_format == "NDHWC" and _is_2byte(src_tensor.dtype)

    def fits(out_lens, sections):
        if not _tile_ap_ok(ks, st, in_spatial, out_lens, 1):
            return False
        item = _tile_bytes(ks, st, pd, in_spatial, out_lens, 1, dtype_bytes, stage_out, partition_limit)
        return item * sections <= total_sbuf

    def fits_interleaved(out_lens):
        return fits(out_lens, _INTERLEAVE_SECTIONS)

    def fits_plain(out_lens):
        return fits(out_lens, 1)

    chunk_axis, chunk = _plan_chunk(out_spatial, fits_interleaved)
    interleaved = fits_interleaved(_tile_lens(out_spatial, chunk_axis, chunk))
    if not interleaved:
        chunk_axis, chunk = _plan_chunk(out_spatial, fits_plain)

    nominal = _tile_lens(out_spatial, chunk_axis, chunk)
    tiles = _spatial_tiles(out_spatial, in_spatial, ks, st, pd, nominal)
    tiled = _is_tiled(out_spatial, nominal)

    item_bytes = _tile_bytes(ks, st, pd, in_spatial, nominal, 1, dtype_bytes, stage_out, partition_limit)

    sections = _INTERLEAVE_SECTIONS if interleaved else 1
    tile_interleave = sections
    load_interleave = sections
    reduce_interleave = _INTERLEAVE_SECTIONS

    group = 1
    if packed and not tiled:
        planes = B * C
        nominal_in = _nominal_in_lens(nominal, in_spatial, ks, st)
        item_vol = nominal_in[0] * nominal_in[1] * nominal_in[2]
        group = max(1, min(max_group_planes, free_elem_budget // max(1, item_vol)))
        if planes > 0 and group > 1:
            group = max(1, min(group, div_ceil(planes, pipeline_tiles) // partition_limit))
        while group > 1 and not (
            _tile_ap_ok(ks, st, in_spatial, nominal, group)
            and _tile_bytes(ks, st, pd, in_spatial, nominal, group, dtype_bytes, stage_out, partition_limit) * sections
            <= total_sbuf
        ):
            group = group // 2

    if packed or C >= partition_limit or C <= 0:
        batch_pack = 1
    else:
        batch_pack = max(1, min(partition_limit // C, B))

    global_items = len(tiles) * _plane_group_count(packed, B, C, group, partition_limit, batch_pack)
    local_items = div_ceil(global_items - prg_id, n_prgs) if prg_id < global_items else 0

    logger = get_logger("max_pooling_3d")
    logger.info(
        "MaxPool3DConfig: "
        "data_format=" + str(data_format) + ", "
        "output_format=" + str(output_format) + ", "
        "B=" + str(B) + ", C=" + str(C) + ", D=" + str(D) + ", H=" + str(H) + ", W=" + str(W) + ", "
        "kernel_size=" + str(ks) + ", stride=" + str(st) + ", padding=" + str(pd) + ", "
        "out_spatial=" + str(out_spatial) + ", dtype=" + str(src_tensor.dtype) + ", "
        "partition_limit=" + str(partition_limit) + ", "
        "chunk_axis=" + str(chunk_axis) + ", chunk=" + str(chunk) + ", "
        "spatial_tiles=" + str(len(tiles)) + ", "
        "tile_bytes=" + str(item_bytes) + "/" + str(total_sbuf) + ", "
        "group=" + str(group) + ", "
        "tile_interleave=" + str(tile_interleave) + ", "
        "load_interleave=" + str(load_interleave) + ", "
        "reduce_interleave=" + str(reduce_interleave) + ", "
        "batch_pack=" + str(batch_pack) + ", "
        "lnc(n_prgs)=" + str(n_prgs) + ", "
        "shard_id(prg_id)=" + str(prg_id) + ", "
        "global_work_items=" + str(global_items) + ", "
        "local_work_items=" + str(local_items)
    )

    return MaxPool3DConfig(
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        ks=ks,
        st=st,
        pd=pd,
        out_spatial=out_spatial,
        dtype=src_tensor.dtype,
        data_format=data_format,
        output_format=output_format,
        packed=packed,
        partition_limit=partition_limit,
        pool_rank=pool_rank,
        group=group,
        evict_group=evict_group,
        tiles=tiles,
        tile_interleave=tile_interleave,
        load_interleave=load_interleave,
        reduce_interleave=reduce_interleave,
        total_sbuf=total_sbuf,
        n_prgs=n_prgs,
        prg_id=prg_id,
        batch_pack=batch_pack,
    )
