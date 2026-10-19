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

"""conv3d_bwd compute kernels: per-gradient strategy implementations (_conv3d_dw_*, _conv3d_dx_*, _conv3d_db) that compose the conv3d_bwd_helpers building blocks; driven by conv3d_bwd."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_helpers import div_ceil
from .conv3d_bwd_helpers import (
    _ACC_DTYPE,
    _ACC_DTYPE_SIZE,
    _NUM_PSUM_BANKS,
    _PSUM_BANK_FMAX,
    _SBUF_HEADROOM,
    Conv3dBwdConfig,
    _build_wt_per_tap,
    _co_chunks,
    _cross_core_reduce_dw,
    _dw_hbm_view,
    _pe_transpose,
    _psum_offsets,
    _pw_dx_contract,
    _pw_dx_store,
    _store_via_result,
    _tap_valid_ranges,
    _valid_output_range,
)


def _sw_alloc(ctx):
    """Transposed filters, identity, live dw accumulators, dx scatter planes, dy/x buffers."""
    ctx.filtersT = []
    for ci_t in range(ctx.n_ci):
        c0i, c1i = ci_t * ctx.ci_tile, min((ci_t + 1) * ctx.ci_tile, ctx.C_in)
        row = []
        for co_t in range(ctx.n_co):
            c0, c1 = co_t * ctx.co_tile, min((co_t + 1) * ctx.co_tile, ctx.C_out)
            ft = nl.ndarray(shape=(c1 - c0, c1i - c0i), dtype=ctx.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=ft, src=ctx.filt2d.slice(dim=0, start=c0i, end=c1i, step=1).slice(dim=1, start=c0, end=c1, step=1)
            )
            row.append(ft)
        ctx.filtersT.append(row)
    ctx.identity = nl.ndarray(shape=(ctx.P_MAX, ctx.P_MAX), dtype=ctx.dtype, buffer=nl.sbuf)
    nl.shared_identity_matrix(ctx.P_MAX, dtype=ctx.dtype, dst=ctx.identity)
    ctx.dw_acc = []
    for ci_t in range(ctx.n_ci):
        if ci_t < ctx.ci_t_lo or ci_t >= ctx.ci_t_hi:
            ctx.dw_acc.append(None)
            continue
        ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
        row = []
        for co_b in range(ctx.n_cb):
            if co_b < ctx.co_b_lo or co_b >= ctx.co_b_hi:
                row.append(None)
                continue
            blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
            row.append(nl.ndarray(shape=(ci_size, blk), dtype=_ACC_DTYPE, buffer=nl.psum))
        ctx.dw_acc.append(row)
    ctx.out_bufs = []
    for _ in range(ctx.cfg.degree("sw_out", 2)):
        buf = nl.ndarray(shape=(ctx.ci_tile, ctx.D, ctx.H, ctx.W), dtype=ctx.dtype, buffer=nl.sbuf)
        nisa.memset(dst=buf[:, :, :, :], value=0.0)
        ctx.out_bufs.append(buf)
    ctx.dx_acc = nl.ndarray(shape=(ctx.ci_tile, ctx.D_out, ctx.H_out, ctx.W_out), dtype=ctx.dtype, buffer=nl.sbuf)
    ctx.dx_acc_flat = ctx.dx_acc.flatten_dims(1, 3)
    ctx.dy_sets, ctx.x_full_sets, ctx.xs_sets = [], [], []
    for _ in range(ctx.degree):
        ctx.dy_sets.append(
            [
                nl.ndarray(
                    shape=(min(ctx.co_tile, ctx.C_out - c * ctx.co_tile), ctx.P_out), dtype=ctx.dtype, buffer=nl.sbuf
                )
                for c in range(ctx.n_co)
            ]
        )
        xf, xs = [], []
        for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
            ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
            xf.append(nl.ndarray(shape=(ci_size, ctx.D, ctx.H, ctx.W), dtype=ctx.dtype, buffer=nl.sbuf))
            xs.append(nl.ndarray(shape=(ci_size, ctx.D_out, ctx.H_out, ctx.W_out), dtype=ctx.dtype, buffer=nl.sbuf))
        ctx.x_full_sets.append(xf)
        ctx.xs_sets.append(xs)


def _pw_alloc(ctx):
    """Declare the resident dy, transposed filters, identity, and dw accumulators."""
    if ctx.dy_double_buf:
        # Two single-batch slots per co tile: dx reads one while the next batch loads into the other.
        ctx.dy_res = [
            [
                nl.ndarray(
                    shape=(min(ctx.co_tile, ctx.C_out - c * ctx.co_tile), ctx.P),
                    dtype=ctx.dtype,
                    buffer=nl.sbuf,
                )
                for c in range(ctx.n_co)
            ]
            for _ in range(2)
        ]
    else:
        ctx.dy_res = [
            nl.ndarray(
                shape=(min(ctx.co_tile, ctx.C_out - c * ctx.co_tile), max(1, ctx.Bp) * ctx.P),
                dtype=ctx.dtype,
                buffer=nl.sbuf,
            )
            for c in range(ctx.n_co)
        ]
    # One transposed-filter tile per co-tile spanning the full C_in on the free axis.
    ctx.filtersT = [
        nl.ndarray(shape=(min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile), ctx.C_in), dtype=ctx.dtype, buffer=nl.sbuf)
        for co_t in range(ctx.n_co)
    ]
    ctx.identity = nl.ndarray(shape=(ctx.P_MAX, ctx.P_MAX), dtype=ctx.dtype, buffer=nl.sbuf)
    nl.shared_identity_matrix(ctx.P_MAX, dtype=ctx.dtype, dst=ctx.identity)
    ctx.dw_psum = [
        [
            nl.ndarray(
                shape=(min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile), min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)),
                dtype=_ACC_DTYPE,
                buffer=nl.psum,
            )
            for cb in range(ctx.n_cb)
        ]
        for ci_t in range(ctx.n_ci)
    ]
    # db accumulated on the Tensor Engine (ones-vector matmul over transposed dy).
    ctx.db_psum = None
    ctx.ones_vec = None
    if ctx.compute_db:
        ctx.db_psum = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.psum)
        ctx.ones_vec = nl.ndarray(shape=(ctx.P_MAX, 1), dtype=ctx.dtype, buffer=nl.sbuf)
        nisa.memset(dst=ctx.ones_vec, value=1.0)


def _pw_dw_psum_tile(ctx, ci_t, co_t, ci_size, co_size):
    """dw accumulator column-range for co tile co_t inside its wide co block."""
    co_start = co_t * ctx.co_tile
    co_b = co_start // ctx.co_blk
    off = co_start - co_b * ctx.co_blk
    return ctx.dw_psum[ci_t][co_b][:ci_size, off : off + co_size]


def _sw_load_operands(ctx, b, slot):
    """dy into dy_sets[slot]; x read contiguous then subsampled onto the output grid in xs."""
    for co_t in range(ctx.n_co):
        c0, c1 = co_t * ctx.co_tile, min((co_t + 1) * ctx.co_tile, ctx.C_out)
        nisa.dma_copy(
            dst=ctx.dy_sets[slot][co_t],
            src=ctx.dy_flat.select(dim=0, index=b).slice(dim=0, start=c0, end=c1, step=1),
            dge_mode=nisa.dge_mode.none,
        )
    for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
        loc = ci_t - ctx.ci_t_lo
        c0i, c1i = ci_t * ctx.ci_tile, min((ci_t + 1) * ctx.ci_tile, ctx.C_in)
        ci_size = c1i - c0i
        nisa.dma_copy(
            dst=ctx.x_full_sets[slot][loc],
            src=ctx.x_in.select(dim=0, index=b).slice(dim=0, start=c0i, end=c1i, step=1),
            dge_mode=nisa.dge_mode.none,
        )
        samp = (
            ctx.x_full_sets[slot][loc]
            .slice(dim=1, start=0, end=ctx.D, step=ctx.sd)
            .slice(dim=2, start=0, end=ctx.H, step=ctx.sh)
            .slice(dim=3, start=0, end=ctx.W, step=ctx.sw)
        )
        nisa.tensor_copy(dst=ctx.xs_sets[slot][loc][:ci_size, :, :, :], src=samp[:ci_size, :, :, :])


def _pw_load_dy_resident(ctx):
    """dy resident once: [co, Bp*P] per co tile, this core's batches side by side on free."""
    spatial = ctx.dy_flat.shape[2]
    coalesce = ctx.dy_res_buf != None and ctx.co_tile == ctx.P_MAX and ctx.C_out % ctx.co_tile == 0 and spatial == ctx.P
    if coalesce:
        for bl in range(ctx.Bp):
            b = ctx.b_lo + bl
            src = ctx.dy_flat.select(dim=0, index=b).ap(
                pattern=[[spatial, ctx.P_MAX], [ctx.P_MAX * spatial, ctx.n_co], [1, spatial]], offset=0
            )
            nisa.dma_copy(dst=ctx.dy_res_buf[:, :, bl * ctx.P : (bl + 1) * ctx.P], src=src, dge_mode=nisa.dge_mode.none)
        return
    for bl in range(ctx.Bp):
        b = ctx.b_lo + bl
        for co_t in range(ctx.n_co):
            c0 = co_t * ctx.co_tile
            c1 = min(c0 + ctx.co_tile, ctx.C_out)
            nisa.dma_copy(
                dst=ctx.dy_res[co_t][:, bl * ctx.P : (bl + 1) * ctx.P],
                src=ctx.dy_flat.select(dim=0, index=b).slice(dim=0, start=c0, end=c1, step=1),
                dge_mode=nisa.dge_mode.none,
            )


def _pw_load_filters_t(ctx):
    """Transpose filters to [co, C_in] per co-tile (batch-independent dx stationary)."""
    for co_t in range(ctx.n_co):
        c0 = co_t * ctx.co_tile
        c1 = min(c0 + ctx.co_tile, ctx.C_out)
        nisa.dma_transpose(dst=ctx.filtersT[co_t], src=ctx.filt2d.slice(dim=1, start=c0, end=c1, step=1))


def _sw_dx(ctx, bl, cur, buf_sel):
    b = ctx.b_lo + bl
    dy_res = ctx.dy_sets[cur]
    for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
        c0i, c1i = ci_t * ctx.ci_tile, min((ci_t + 1) * ctx.ci_tile, ctx.C_in)
        ci_size = c1i - c0i
        for p_off in range(0, ctx.P_out, ctx.F_MAX):
            p_size = min(ctx.F_MAX, ctx.P_out - p_off)
            dx_psum = nl.ndarray(shape=(ci_size, p_size), dtype=_ACC_DTYPE, buffer=nl.psum)
            for co_t in range(ctx.n_co):
                co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
                nisa.nc_matmul(
                    dst=dx_psum,
                    stationary=ctx.filtersT[ci_t][co_t][:co_size, :ci_size],
                    moving=dy_res[co_t][:co_size, p_off : p_off + p_size],
                )
            nisa.tensor_copy(
                dst=ctx.dx_acc_flat[:ci_size, p_off : p_off + p_size], src=dx_psum, engine=nisa.scalar_engine
            )
        out_buf = ctx.out_bufs[buf_sel]
        grid = (
            out_buf.slice(dim=1, start=0, end=ctx.D, step=ctx.sd)
            .slice(dim=2, start=0, end=ctx.H, step=ctx.sh)
            .slice(dim=3, start=0, end=ctx.W, step=ctx.sw)
        )
        nisa.tensor_copy(dst=grid[:ci_size, :, :, :], src=ctx.dx_acc[:ci_size, :, :, :])
        if not (ctx.shard_co and ctx.prg_id != 0):
            nisa.dma_copy(
                dst=ctx.dx.select(dim=0, index=b).slice(dim=0, start=c0i, end=c1i, step=1),
                src=out_buf[:ci_size, :, :, :],
                dge_mode=nisa.dge_mode.none,
            )
        buf_sel = (buf_sel + 1) % len(ctx.out_bufs)
    return buf_sel


def _sw_dw(ctx, bl, cur):
    dy_res = ctx.dy_sets[cur]
    x_samp = ctx.xs_sets[cur]
    dyt_tiles, xt_tiles = [], []
    for pt in range(ctx.n_pos_tiles):
        p_off = pt * ctx.P_MAX
        p_size = min(ctx.P_MAX, ctx.P_out - p_off)
        dyt = nl.ndarray(shape=(p_size, ctx.C_out), dtype=ctx.dtype, buffer=nl.sbuf)
        # Pack up to F_MAX (512) worth of co-tiles into one PSUM bank and drain the bank once.
        for bank0 in range(0, ctx.C_out, ctx.F_MAX):
            bank_w = min(ctx.F_MAX, ctx.C_out - bank0)
            dyt_psum = nl.ndarray(shape=(p_size, bank_w), dtype=_ACC_DTYPE, buffer=nl.psum)
            for c0 in range(bank0, bank0 + bank_w, ctx.co_tile):
                co_t = c0 // ctx.co_tile
                co_size = min(ctx.co_tile, ctx.C_out - c0)
                nisa.nc_matmul(
                    dst=dyt_psum[:p_size, c0 - bank0 : c0 - bank0 + co_size],
                    stationary=dy_res[co_t][:co_size, p_off : p_off + p_size],
                    moving=ctx.identity[:co_size, :co_size],
                    accumulate=False,
                )
            nisa.tensor_copy(dst=dyt[:p_size, bank0 : bank0 + bank_w], src=dyt_psum, engine=nisa.scalar_engine)
        dyt_tiles.append(dyt)
        xt_row = []
        for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
            ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
            xs_flat = x_samp[ci_t - ctx.ci_t_lo].flatten_dims(1, 3)
            xt = nl.ndarray(shape=(p_size, ci_size), dtype=ctx.dtype, buffer=nl.sbuf)
            xt_psum = nl.ndarray(shape=(p_size, ci_size), dtype=_ACC_DTYPE, buffer=nl.psum)
            nisa.nc_matmul(
                dst=xt_psum,
                stationary=xs_flat[:ci_size, p_off : p_off + p_size],
                moving=ctx.identity[:ci_size, :ci_size],
                accumulate=False,
            )
            nisa.tensor_copy(dst=xt[:p_size, :ci_size], src=xt_psum, engine=nisa.scalar_engine)
            xt_row.append(xt)
        xt_tiles.append(xt_row)
    for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
        ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
        for co_b in range(ctx.co_b_lo, ctx.co_b_hi):
            b0 = co_b * ctx.co_blk
            blk = min(ctx.co_blk, ctx.C_out - b0)
            for pt in range(ctx.n_pos_tiles):
                p_size = min(ctx.P_MAX, ctx.P_out - pt * ctx.P_MAX)
                nisa.nc_matmul(
                    dst=ctx.dw_acc[ci_t][co_b][:ci_size, :blk],
                    stationary=xt_tiles[pt][ci_t - ctx.ci_t_lo][:p_size, :ci_size],
                    moving=dyt_tiles[pt][:p_size, b0 : b0 + blk],
                )


def _pw_dx(ctx):
    """dx[b,ci,p] = sum_co W[ci,co]*dy[b,co,p]."""
    for bl in range(ctx.Bp):
        b = ctx.b_lo + bl
        for ci_t in range(ctx.n_ci):
            c0i = ci_t * ctx.ci_tile
            ci_size = min(ctx.ci_tile, ctx.C_in - c0i)
            for p_off in range(0, ctx.P, ctx.F_MAX):
                p_size = min(ctx.F_MAX, ctx.P - p_off)
                dx_psum = _pw_dx_contract(ctx, ctx.dy_res, bl * ctx.P, c0i, ci_size, p_off, p_size)
                _pw_dx_store(ctx, dx_psum, b, c0i, ci_size, p_off, p_size)


def _pw_dw(ctx):
    """dw[ci,co] = sum_{b,p} x[b,ci,p]*dy[b,co,p]."""
    if ctx.pw_coalesce_xt:
        _pw_dw_wide(ctx)
    else:
        _pw_dw_perbatch(ctx)


def _pw_dw_perbatch(ctx):
    """dw[ci,co] = sum_{b,p} x[b,ci,p]*dy[b,co,p], accumulated live in dw_psum."""
    xt_all = nl.ndarray(shape=(ctx.P_MAX, ctx.C_in), dtype=ctx.dtype, buffer=nl.sbuf)
    dyt_bufs = [
        nl.ndarray(shape=(ctx.P_MAX, min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)), dtype=ctx.dtype, buffer=nl.sbuf)
        for cb in range(ctx.n_cb)
    ]
    for bl in range(ctx.Bp):
        b = ctx.b_lo + bl
        for p_off in range(0, ctx.P, ctx.P_MAX):
            p_size = min(ctx.P_MAX, ctx.P - p_off)
            for co_b in range(ctx.n_cb):
                blk0 = co_b * ctx.co_blk
                blk = min(ctx.co_blk, ctx.C_out - blk0)
                if ctx.dw_dy_via_dma:
                    nisa.dma_transpose(
                        dst=dyt_bufs[co_b][:p_size, :blk],
                        src=ctx.dy_flat.select(dim=0, index=b)
                        .slice(dim=0, start=blk0, end=blk0 + blk, step=1)
                        .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                    )
                else:
                    dyt_psum = nl.ndarray(shape=(p_size, blk), dtype=_ACC_DTYPE, buffer=nl.psum)
                    for ct_local in range(div_ceil(blk, ctx.co_tile)):
                        co_t = co_b * ctx.co_tiles_per_blk + ct_local
                        c0 = co_t * ctx.co_tile
                        co_size = min(ctx.co_tile, ctx.C_out - c0)
                        off = c0 - blk0
                        nisa.nc_matmul(
                            dst=dyt_psum[:, off : off + co_size],
                            stationary=ctx.dy_res[co_t][:co_size, bl * ctx.P + p_off : bl * ctx.P + p_off + p_size],
                            moving=ctx.identity[:co_size, :co_size],
                            accumulate=False,
                        )
                    nisa.tensor_copy(
                        dst=dyt_bufs[co_b][:p_size, :blk],
                        src=dyt_psum,
                        engine=nisa.scalar_engine if co_b % 2 == 0 else nisa.vector_engine,
                    )
            nisa.dma_transpose(
                dst=xt_all[:p_size, : ctx.C_in],
                src=ctx.x_flat.select(dim=0, index=b).slice(dim=1, start=p_off, end=p_off + p_size, step=1),
            )
            for ci_t in range(ctx.n_ci):
                c0i = ci_t * ctx.ci_tile
                c1i = min(c0i + ctx.ci_tile, ctx.C_in)
                for co_b in range(ctx.n_cb):
                    blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
                    nisa.nc_matmul(
                        dst=ctx.dw_psum[ci_t][co_b],
                        stationary=xt_all[:p_size, c0i:c1i],
                        moving=dyt_bufs[co_b][:p_size, :blk],
                    )


def _pw_dw_wide(ctx):
    """dw[ci,co] = sum_{b,p} x[b,ci,p]*dy[b,co,p], accumulated live in dw_psum."""
    ncols_x = max(1, ctx.Bp) * ctx.C_in
    xt_all = nl.ndarray(shape=(ctx.P_MAX, ncols_x), dtype=ctx.dtype, buffer=nl.sbuf)
    x2d = ctx.x_flat.flatten_dims(0, 1)  # [B*C_in, P]
    dyt_bufs = [
        nl.ndarray(shape=(ctx.P_MAX, min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)), dtype=ctx.dtype, buffer=nl.sbuf)
        for cb in range(ctx.n_cb)
    ]
    for p_off in range(0, ctx.P, ctx.P_MAX):
        p_size = min(ctx.P_MAX, ctx.P - p_off)
        # Bring x into positions-major xt_all[p, (bl,ci)]; coalesce all batches for large planes.
        if ctx.pw_coalesce_xt:
            nisa.dma_transpose(
                dst=xt_all[:p_size, : ctx.Bp * ctx.C_in],
                src=x2d.slice(dim=0, start=ctx.b_lo * ctx.C_in, end=ctx.b_hi * ctx.C_in, step=1).slice(
                    dim=1, start=p_off, end=p_off + p_size, step=1
                ),
            )
        else:
            for bl in range(ctx.Bp):
                b = ctx.b_lo + bl
                nisa.dma_transpose(
                    dst=xt_all[:p_size, bl * ctx.C_in : (bl + 1) * ctx.C_in],
                    src=ctx.x_flat.select(dim=0, index=b).slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                )
        if ctx.dw_dy_via_dma:
            for bl in range(ctx.Bp):
                b = ctx.b_lo + bl
                for co_b in range(ctx.n_cb):
                    blk0 = co_b * ctx.co_blk
                    blk = min(ctx.co_blk, ctx.C_out - blk0)
                    nisa.dma_transpose(
                        dst=dyt_bufs[co_b][:p_size, :blk],
                        src=ctx.dy_flat.select(dim=0, index=b)
                        .slice(dim=0, start=blk0, end=blk0 + blk, step=1)
                        .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                    )
                xcol0 = bl * ctx.C_in
                for ci_t in range(ctx.n_ci):
                    c0i = ci_t * ctx.ci_tile
                    c1i = min(c0i + ctx.ci_tile, ctx.C_in)
                    for co_b in range(ctx.n_cb):
                        blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
                        nisa.nc_matmul(
                            dst=ctx.dw_psum[ci_t][co_b],
                            stationary=xt_all[:p_size, xcol0 + c0i : xcol0 + c1i],
                            moving=dyt_bufs[co_b][:p_size, :blk],
                        )
            continue
        # PE-transpose path: dy is already resident in ctx.dy_res (SBUF); transpose it on the PE.
        for co_b in range(ctx.n_cb):
            blk0 = co_b * ctx.co_blk
            blk = min(ctx.co_blk, ctx.C_out - blk0)
            grp = max(1, ctx.F_MAX // blk) if ctx.pw_coalesce_xt else 1
            grp_idx = 0
            for bl0 in range(0, ctx.Bp, grp):
                gc = min(grp, ctx.Bp - bl0)
                wide = gc * blk
                dyt_psum = nl.ndarray(shape=(p_size, wide), dtype=_ACC_DTYPE, buffer=nl.psum)
                for j in range(gc):
                    bl = bl0 + j
                    for ct_local in range(div_ceil(blk, ctx.co_tile)):
                        co_t = co_b * ctx.co_tiles_per_blk + ct_local
                        c0 = co_t * ctx.co_tile
                        co_size = min(ctx.co_tile, ctx.C_out - c0)
                        off = c0 - blk0
                        nisa.nc_matmul(
                            dst=dyt_psum[:, j * blk + off : j * blk + off + co_size],
                            stationary=ctx.dy_res[co_t][:co_size, bl * ctx.P + p_off : bl * ctx.P + p_off + p_size],
                            moving=ctx.identity[:co_size, :co_size],
                            accumulate=False,
                        )
                dyt_wide = nl.ndarray(shape=(p_size, wide), dtype=ctx.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=dyt_wide,
                    src=dyt_psum,
                    engine=nisa.scalar_engine,
                )
                grp_idx += 1
                for j in range(gc):
                    bl = bl0 + j
                    xcol0 = bl * ctx.C_in
                    for ci_t in range(ctx.n_ci):
                        c0i = ci_t * ctx.ci_tile
                        c1i = min(c0i + ctx.ci_tile, ctx.C_in)
                        nisa.nc_matmul(
                            dst=ctx.dw_psum[ci_t][co_b],
                            stationary=xt_all[:p_size, xcol0 + c0i : xcol0 + c1i],
                            moving=dyt_wide[:p_size, j * blk : j * blk + blk],
                        )


def _pw_dw_setup(ctx):
    """Allocate the (batch-independent) SBUF/PSUM scratch reused across every dw batch."""
    xt_all = nl.ndarray(shape=(ctx.P_MAX, ctx.C_in), dtype=ctx.dtype, buffer=nl.sbuf)
    x_via_pe = ctx.C_in <= ctx.P_MAX
    x_all_sb = xt_psum = None
    if x_via_pe:
        x_all_sb = nl.ndarray(shape=(ctx.C_in, ctx.P), dtype=ctx.dtype, buffer=nl.sbuf)
        xt_psum = nl.ndarray(shape=(ctx.P_MAX, ctx.C_in), dtype=_ACC_DTYPE, buffer=nl.psum)
    dyt_bufs = [
        nl.ndarray(shape=(ctx.P_MAX, min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)), dtype=ctx.dtype, buffer=nl.sbuf)
        for cb in range(ctx.n_cb)
    ]
    return {"xt_all": xt_all, "x_via_pe": x_via_pe, "x_all_sb": x_all_sb, "xt_psum": xt_psum, "dyt_bufs": dyt_bufs}


def _pw_load_dy_batch(ctx, slot, b):
    """Load one batch's dy into double-buffer slot: [co, P] per co tile."""
    for co_t in range(ctx.n_co):
        c0 = co_t * ctx.co_tile
        c1 = min(c0 + ctx.co_tile, ctx.C_out)
        nisa.dma_copy(
            dst=ctx.dy_res[slot][co_t][:, : ctx.P],
            src=ctx.dy_flat.select(dim=0, index=b).slice(dim=0, start=c0, end=c1, step=1),
            dge_mode=nisa.dge_mode.none,
        )


def _pw_dx_batch(ctx, bl, cur):
    """dx for one batch, reading the double-buffered dy slot `cur` (single-batch [co, P])."""
    b = ctx.b_lo + bl
    for ci_t in range(ctx.n_ci):
        c0i = ci_t * ctx.ci_tile
        ci_size = min(ctx.ci_tile, ctx.C_in - c0i)
        for p_off in range(0, ctx.P, ctx.F_MAX):
            p_size = min(ctx.F_MAX, ctx.P - p_off)
            dx_psum = _pw_dx_contract(ctx, ctx.dy_res[cur], 0, c0i, ci_size, p_off, p_size)
            _pw_dx_store(ctx, dx_psum, b, c0i, ci_size, p_off, p_size)


def _pw_dw_batch(ctx, bl, bufs, cur=None):
    """dw contribution of one batch, accumulated live into ctx.dw_psum (first batch resets)."""
    b = ctx.b_lo + bl
    xt_all, dyt_bufs = bufs["xt_all"], bufs["dyt_bufs"]
    x_via_pe, x_all_sb, xt_psum = bufs["x_via_pe"], bufs["x_all_sb"], bufs["xt_psum"]
    if x_via_pe:
        nisa.dma_copy(
            dst=x_all_sb[: ctx.C_in, : ctx.P],
            src=ctx.x_flat.select(dim=0, index=b),
        )
    for p_off in range(0, ctx.P, ctx.P_MAX):
        p_size = min(ctx.P_MAX, ctx.P - p_off)
        for co_b in range(ctx.n_cb):
            blk0 = co_b * ctx.co_blk
            blk = min(ctx.co_blk, ctx.C_out - blk0)
            if cur != None:
                # PE transpose off the resident double-buffered dy slot (frees the DMA queue).
                dyt_psum = nl.ndarray(shape=(p_size, blk), dtype=_ACC_DTYPE, buffer=nl.psum)
                for ct_local in range(div_ceil(blk, ctx.co_tile)):
                    co_t = co_b * ctx.co_tiles_per_blk + ct_local
                    c0 = co_t * ctx.co_tile
                    co_size = min(ctx.co_tile, ctx.C_out - c0)
                    off = c0 - blk0
                    nisa.nc_matmul(
                        dst=dyt_psum[:, off : off + co_size],
                        stationary=ctx.dy_res[cur][co_t][:co_size, p_off : p_off + p_size],
                        moving=ctx.identity[:co_size, :co_size],
                        accumulate=False,
                    )
                nisa.tensor_copy(
                    dst=dyt_bufs[co_b][:p_size, :blk],
                    src=dyt_psum,
                    engine=nisa.scalar_engine if co_b % 2 == 0 else nisa.vector_engine,
                )
            elif ctx.dw_dy_via_dma:
                nisa.dma_transpose(
                    dst=dyt_bufs[co_b][:p_size, :blk],
                    src=ctx.dy_flat.select(dim=0, index=b)
                    .slice(dim=0, start=blk0, end=blk0 + blk, step=1)
                    .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                )
            else:
                dyt_psum = nl.ndarray(shape=(p_size, blk), dtype=_ACC_DTYPE, buffer=nl.psum)
                for ct_local in range(div_ceil(blk, ctx.co_tile)):
                    co_t = co_b * ctx.co_tiles_per_blk + ct_local
                    c0 = co_t * ctx.co_tile
                    co_size = min(ctx.co_tile, ctx.C_out - c0)
                    off = c0 - blk0
                    nisa.nc_matmul(
                        dst=dyt_psum[:, off : off + co_size],
                        stationary=ctx.dy_res[co_t][:co_size, bl * ctx.P + p_off : bl * ctx.P + p_off + p_size],
                        moving=ctx.identity[:co_size, :co_size],
                        accumulate=False,
                    )
                nisa.tensor_copy(
                    dst=dyt_bufs[co_b][:p_size, :blk],
                    src=dyt_psum,
                    engine=nisa.scalar_engine if co_b % 2 == 0 else nisa.vector_engine,
                )
            if ctx.compute_db:
                # db[co] += sum_p dyt[p, co], folded onto PE via a ones moving vector.
                for ct_local in range(div_ceil(blk, ctx.co_tile)):
                    co_t = co_b * ctx.co_tiles_per_blk + ct_local
                    off = co_t * ctx.co_tile - blk0
                    co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
                    nisa.nc_matmul(
                        dst=ctx.db_psum[:co_size, co_t : co_t + 1],
                        stationary=dyt_bufs[co_b][:p_size, off : off + co_size],
                        moving=ctx.ones_vec[:p_size, :1],
                    )
        if x_via_pe:
            nisa.nc_matmul(
                dst=xt_psum[:p_size, : ctx.C_in],
                stationary=x_all_sb[: ctx.C_in, p_off : p_off + p_size],
                moving=ctx.identity[: ctx.C_in, : ctx.C_in],
                accumulate=False,
            )
            nisa.tensor_copy(dst=xt_all[:p_size, : ctx.C_in], src=xt_psum[:p_size, : ctx.C_in])
        else:
            nisa.dma_transpose(
                dst=xt_all[:p_size, : ctx.C_in],
                src=ctx.x_flat.select(dim=0, index=b).slice(dim=1, start=p_off, end=p_off + p_size, step=1),
            )
        for ci_t in range(ctx.n_ci):
            c0i = ci_t * ctx.ci_tile
            c1i = min(c0i + ctx.ci_tile, ctx.C_in)
            for co_b in range(ctx.n_cb):
                blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
                nisa.nc_matmul(
                    dst=ctx.dw_psum[ci_t][co_b],
                    stationary=xt_all[:p_size, c0i:c1i],
                    moving=dyt_bufs[co_b][:p_size, :blk],
                )


def _pw_dx_dw_fused(ctx):
    """Double-buffered dx+dw: prefetch batch bl+1's dy while dw of batch bl runs on the PE."""
    bufs = _pw_dw_setup(ctx)
    if ctx.Bp > 0:
        _pw_load_dy_batch(ctx, 0, ctx.b_lo)
    for bl in range(ctx.Bp):
        cur = bl % 2
        _pw_dx_batch(ctx, bl, cur)
        if bl + 1 < ctx.Bp:
            _pw_load_dy_batch(ctx, (bl + 1) % 2, ctx.b_lo + bl + 1)  # prefetch next batch
        _pw_dw_batch(ctx, bl, bufs, cur)


def _pw_db(ctx):
    """Produce db_part[co] for _pw_store_dw_db."""
    ctx.db_part = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)
    for co_t in range(ctx.n_co):
        co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
        if not ctx.dw_has_work:
            nisa.memset(dst=ctx.db_part[:co_size, co_t : co_t + 1], value=0.0)
        elif ctx.dy_double_buf:
            nisa.tensor_copy(
                dst=ctx.db_part[:co_size, co_t : co_t + 1],
                src=ctx.db_psum[:co_size, co_t : co_t + 1],
            )
        else:
            nisa.tensor_reduce(
                dst=ctx.db_part[:co_size, co_t : co_t + 1],
                data=ctx.dy_res[co_t][:co_size, : ctx.Bp * ctx.P],
                op=nl.add,
                axis=1,
                keepdims=True,
            )


def _sw_store_dw(ctx):
    def acc_tile(ci_t, c0, c1, ci_size):
        co_b = c0 // ctx.co_blk
        off = c0 - co_b * ctx.co_blk
        return ctx.dw_acc[ci_t][co_b][:ci_size, off : off + (c1 - c0)]

    if not ctx.reduce_across:
        for ci_t in range(ctx.ci_t_lo, ctx.ci_t_hi):
            c0i, c1i = ci_t * ctx.ci_tile, min((ci_t + 1) * ctx.ci_tile, ctx.C_in)
            ci_size = c1i - c0i
            for co_t in range(ctx.n_co):
                c0, c1 = co_t * ctx.co_tile, min((co_t + 1) * ctx.co_tile, ctx.C_out)
                if not (ctx.co_b_lo * ctx.co_blk <= c0 < ctx.co_b_hi * ctx.co_blk):
                    continue
                dw_sbuf = nl.ndarray(shape=(ci_size, c1 - c0), dtype=ctx.dtype, buffer=nl.sbuf)
                if ctx.dw_has_work:
                    nisa.tensor_copy(dst=dw_sbuf, src=acc_tile(ci_t, c0, c1, ci_size))
                else:
                    nisa.memset(dst=dw_sbuf, value=0.0)
                nisa.dma_copy(
                    dst=_dw_hbm_view(ctx.dw, 0, 0, 0, c0i, c1i, c0, c1), src=dw_sbuf, dge_mode=nisa.dge_mode.none
                )
        return
    dw_part = nl.ndarray(shape=(ctx.ci_tile, ctx.n_ci * ctx.C_out), dtype=_ACC_DTYPE, buffer=nl.sbuf)
    for ci_t in range(ctx.n_ci):
        ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
        if not ctx.dw_has_work:
            nisa.memset(dst=dw_part[:ci_size, ci_t * ctx.C_out : (ci_t + 1) * ctx.C_out], value=0.0)
            continue
        for co_b in range(ctx.n_cb):
            b0 = co_b * ctx.co_blk
            blk = min(ctx.co_blk, ctx.C_out - b0)
            nisa.tensor_copy(
                dst=dw_part[:ci_size, ci_t * ctx.C_out + b0 : ci_t * ctx.C_out + b0 + blk],
                src=ctx.dw_acc[ci_t][co_b][:ci_size, :blk],
            )
    _cross_core_reduce_dw(ctx, dw_part, lambda ci_t, co_t: ci_t * ctx.C_out + co_t * ctx.co_tile)


def _pw_store_dw_db(ctx):
    if not ctx.reduce_across:
        for ci_t in range(ctx.n_ci):
            c0i = ci_t * ctx.ci_tile
            ci_size = min(ctx.ci_tile, ctx.C_in - c0i)
            for co_t in range(ctx.n_co):
                c0 = co_t * ctx.co_tile
                co_size = min(ctx.co_tile, ctx.C_out - c0)
                dw_sbuf = nl.ndarray(shape=(ci_size, co_size), dtype=ctx.dtype, buffer=nl.sbuf)
                if ctx.dw_has_work:
                    nisa.tensor_copy(dst=dw_sbuf, src=_pw_dw_psum_tile(ctx, ci_t, co_t, ci_size, co_size))
                else:
                    nisa.memset(dst=dw_sbuf, value=0.0)
                nisa.dma_copy(
                    dst=_dw_hbm_view(ctx.dw, 0, 0, 0, c0i, c0i + ci_size, c0, c0 + co_size),
                    src=dw_sbuf,
                    dge_mode=nisa.dge_mode.none,
                )
        if ctx.compute_db:
            db_sbuf = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=ctx.dy.dtype, buffer=nl.sbuf)
            for co_t in range(ctx.n_co):
                c0 = co_t * ctx.co_tile
                co_size = min(ctx.co_tile, ctx.C_out - c0)
                nisa.tensor_copy(dst=db_sbuf[:co_size, co_t : co_t + 1], src=ctx.db_part[:co_size, co_t : co_t + 1])
                nisa.dma_copy(
                    dst=ctx.db.slice(dim=0, start=c0, end=c0 + co_size, step=1),
                    src=db_sbuf[:co_size, co_t],
                    dge_mode=nisa.dge_mode.none,
                )
        return
    other = 1 - ctx.prg_id
    # dw cross-core exchange+reduce+store; skipped for the PACKED unit variant (done inline there).
    if ctx.pw_store_dw_here:
        slots = ctx.n_ci * ctx.n_co
        dw_part = nl.ndarray(shape=(ctx.P_MAX, slots * ctx.co_tile), dtype=_ACC_DTYPE, buffer=nl.sbuf)
        for ci_t in range(ctx.n_ci):
            ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
            for co_t in range(ctx.n_co):
                co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
                slot = ci_t * ctx.n_co + co_t
                dst = dw_part[:ci_size, slot * ctx.co_tile : slot * ctx.co_tile + co_size]
                if ctx.dw_has_work:
                    nisa.tensor_copy(dst=dst, src=_pw_dw_psum_tile(ctx, ci_t, co_t, ci_size, co_size))
                else:
                    nisa.memset(dst=dst, value=0.0)
        _cross_core_reduce_dw(ctx, dw_part, lambda ci_t, co_t: (ci_t * ctx.n_co + co_t) * ctx.co_tile)
    if ctx.compute_db:
        db_recv = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)
        nisa.sendrecv(src=ctx.db_part, dst=db_recv, send_to_rank=other, recv_from_rank=other, pipe_id=1)
    if ctx.compute_db and ctx.prg_id == 0:
        db_red = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)
        db_out = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=ctx.dy.dtype, buffer=nl.sbuf)
        for co_t in range(ctx.n_co):
            c0 = co_t * ctx.co_tile
            co_size = min(ctx.co_tile, ctx.C_out - c0)
            nisa.tensor_tensor(
                dst=db_red[:co_size, co_t : co_t + 1],
                data1=ctx.db_part[:co_size, co_t : co_t + 1],
                data2=db_recv[:co_size, co_t : co_t + 1],
                op=nl.add,
            )
            nisa.tensor_copy(dst=db_out[:co_size, co_t : co_t + 1], src=db_red[:co_size, co_t : co_t + 1])
            nisa.dma_copy(
                dst=ctx.db.slice(dim=0, start=c0, end=c0 + co_size, step=1),
                src=db_out[:co_size, co_t],
                dge_mode=nisa.dge_mode.none,
            )


def _pw_alloc_pk(ctx):
    """Declare the resident dy, transposed filters, identity, and dw accumulators (PACKED variant)."""
    # Single contiguous resident-dy buffer so the per-batch load can be coalesced into one DMA.
    ctx.dy_res_buf = nl.ndarray(
        shape=(ctx.P_MAX, ctx.n_co, max(1, ctx.Bp) * ctx.P),
        dtype=ctx.dtype,
        buffer=nl.sbuf,
    )
    ctx.dy_res = [ctx.dy_res_buf[:, c, :] for c in range(ctx.n_co)]
    # One transposed-filter tile per co-tile with the full C_in span on the free axis.
    ctx.filtersT = []
    for co_t in range(ctx.n_co):
        co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
        ctx.filtersT.append(nl.ndarray(shape=(co_size, ctx.C_in), dtype=ctx.dtype, buffer=nl.sbuf))
    ctx.identity = nl.ndarray(shape=(ctx.P_MAX, ctx.P_MAX), dtype=ctx.dtype, buffer=nl.sbuf)
    nl.shared_identity_matrix(ctx.P_MAX, dtype=ctx.dtype, dst=ctx.identity)
    # dw accumulators live in SBUF (fp32) so the full n_ci x n_cb grid can stay resident.
    ctx.dw_sbuf = [
        [
            nl.ndarray(
                shape=(min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile), min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)),
                dtype=_ACC_DTYPE,
                buffer=nl.sbuf,
            )
            for cb in range(ctx.n_cb)
        ]
        for ci_t in range(ctx.n_ci)
    ]
    # Double-buffered send/recv scratch so each co-block's cross-shard exchange overlaps compute.
    if ctx.reduce_across:
        ctx.dw_buf_depth = 2
        ctx.dw_send = [
            nl.ndarray(shape=(ctx.P_MAX, ctx.n_ci * ctx.co_blk), dtype=ctx.dtype, buffer=nl.sbuf)
            for _ in range(ctx.dw_buf_depth)
        ]
        ctx.dw_recv = [
            nl.ndarray(shape=(ctx.P_MAX, ctx.n_ci * ctx.co_blk), dtype=ctx.dtype, buffer=nl.sbuf)
            for _ in range(ctx.dw_buf_depth)
        ]


def _pw_coblk_reduce_store_pk(ctx, co_b, buf):
    """Reduce this co-block's dw across the two shards and store it."""
    other = 1 - ctx.prg_id
    blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
    c0 = co_b * ctx.co_blk
    nisa.sendrecv(src=ctx.dw_send[buf], dst=ctx.dw_recv[buf], send_to_rank=other, recv_from_rank=other, pipe_id=buf)
    rper = div_ceil(ctx.n_ci, ctx.n_prgs)
    for ci_t in range(min(rper * ctx.prg_id, ctx.n_ci), min(rper * ctx.prg_id + rper, ctx.n_ci)):
        c0i = ci_t * ctx.ci_tile
        ci_size = min(ctx.ci_tile, ctx.C_in - c0i)
        base = ci_t * ctx.co_blk
        # Fuse the cross-shard add and the dtype cast into the DMA store on the DMA engine.
        nisa.dma_compute(
            dst=_dw_hbm_view(ctx.dw, 0, 0, 0, c0i, c0i + ci_size, c0, c0 + blk),
            srcs=[
                ctx.dw_send[buf][:ci_size, base : base + blk],
                ctx.dw_recv[buf][:ci_size, base : base + blk],
            ],
            reduce_op=nl.add,
        )


def _pw_dw_pk(ctx):
    """dw[ci,co] = sum_{b,p} x[b,ci,p]*dy[b,co,p]."""
    # position tiles per batch (P_MAX cap on the contraction axis)
    pos_tiles = [(p_off, min(ctx.P_MAX, ctx.P - p_off)) for p_off in range(0, ctx.P, ctx.P_MAX)]

    # db bias-gradient reduction is interleaved into phase 2 below (per co-block).
    if ctx.compute_db:
        ctx.db_part = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)

    # Pack multiple batches' positions into a single contraction tile (up to P_MAX rows).
    bpack = max(1, ctx.P_MAX // ctx.P) if len(pos_tiles) == 1 else 1

    # Build slot descriptors: each slot is a list of (bl, p_off, p_size, row_off) segments.
    slots = []
    for p_off, p_size in pos_tiles:
        bl = 0
        while bl < ctx.Bp:
            nb = min(bpack, ctx.Bp - bl)
            segs = []
            row = 0
            for k in range(nb):
                segs.append((bl + k, p_off, p_size, row))
                row += p_size
            slots.append((segs, row))
            bl += nb
    n_slots = len(slots)

    # Phase 1: resident transposed operands, one entry per slot (batch-group x position-tile).
    xt_res = [nl.ndarray(shape=(ctx.P_MAX, ctx.C_in), dtype=ctx.dtype, buffer=nl.sbuf) for _ in range(n_slots)]
    dyt_res = [
        [
            nl.ndarray(shape=(ctx.P_MAX, min(ctx.co_blk, ctx.C_out - cb * ctx.co_blk)), dtype=ctx.dtype, buffer=nl.sbuf)
            for cb in range(ctx.n_cb)
        ]
        for _ in range(n_slots)
    ]
    for si in range(n_slots):
        segs, total_rows = slots[si]
        bl0, p_off0, _, _ = segs[0]
        for co_b in range(ctx.n_cb):
            blk0 = co_b * ctx.co_blk
            blk = min(ctx.co_blk, ctx.C_out - blk0)
            if ctx.dw_dy_via_dma:
                for bl, p_off, p_size, row in segs:
                    b = ctx.b_lo + bl
                    nisa.dma_transpose(
                        dst=dyt_res[si][co_b][row : row + p_size, :blk],
                        src=ctx.dy_flat.select(dim=0, index=b)
                        .slice(dim=0, start=blk0, end=blk0 + blk, step=1)
                        .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                    )
            else:
                # Segments are contiguous in dy_res columns (consecutive batches, same p_off).
                col0 = bl0 * ctx.P + p_off0
                dyt_psum = nl.ndarray(shape=(total_rows, blk), dtype=_ACC_DTYPE, buffer=nl.psum)
                for ct_local in range(div_ceil(blk, ctx.co_tile)):
                    co_t = co_b * ctx.co_tiles_per_blk + ct_local
                    c0 = co_t * ctx.co_tile
                    co_size = min(ctx.co_tile, ctx.C_out - c0)
                    off = c0 - blk0
                    nisa.nc_matmul(
                        dst=dyt_psum[:, off : off + co_size],
                        stationary=ctx.dy_res[co_t][:co_size, col0 : col0 + total_rows],
                        moving=ctx.identity[:co_size, :co_size],
                        accumulate=False,
                    )
                nisa.tensor_copy(
                    dst=dyt_res[si][co_b][:total_rows, :blk],
                    src=dyt_psum,
                    engine=nisa.scalar_engine if co_b % 2 == 0 else nisa.vector_engine,
                )
        if bpack == 1:
            for bl, p_off, p_size, row in segs:
                b = ctx.b_lo + bl
                nisa.dma_transpose(
                    dst=xt_res[si][row : row + p_size, : ctx.C_in],
                    src=ctx.x_flat.select(dim=0, index=b).slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                )
        else:
            # dma_transpose cannot write to a nonzero starting partition, so PE-transpose x instead.
            for ci_t in range(ctx.n_ci):
                c0i = ci_t * ctx.ci_tile
                c1i = min(c0i + ctx.ci_tile, ctx.C_in)
                ci_size = c1i - c0i
                x_ld = nl.ndarray(shape=(ci_size, total_rows), dtype=ctx.dtype, buffer=nl.sbuf)
                for bl, p_off, p_size, row in segs:
                    b = ctx.b_lo + bl
                    nisa.dma_copy(
                        dst=x_ld[:ci_size, row : row + p_size],
                        src=ctx.x_flat.select(dim=0, index=b)
                        .slice(dim=0, start=c0i, end=c1i, step=1)
                        .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
                        dge_mode=nisa.dge_mode.none,
                    )
                xt_psum = nl.ndarray(shape=(total_rows, ci_size), dtype=_ACC_DTYPE, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=xt_psum,
                    stationary=x_ld[:ci_size, :total_rows],
                    moving=ctx.identity[:ci_size, :ci_size],
                    accumulate=False,
                )
                nisa.tensor_copy(dst=xt_res[si][:total_rows, c0i:c1i], src=xt_psum)

    # Phase 2: accumulate one co-block at a time entirely in PSUM (only n_ci banks live).
    for co_b in range(ctx.n_cb):
        blk = min(ctx.co_blk, ctx.C_out - co_b * ctx.co_blk)
        # Interleave the db reduction for the co-tiles owned by this co-block with the PE work.
        if ctx.compute_db:
            for ct_local in range(ctx.co_tiles_per_blk):
                co_t = co_b * ctx.co_tiles_per_blk + ct_local
                if co_t >= ctx.n_co:
                    break
                co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
                if ctx.dw_has_work:
                    nisa.tensor_reduce(
                        dst=ctx.db_part[:co_size, co_t : co_t + 1],
                        data=ctx.dy_res[co_t][:co_size, : ctx.Bp * ctx.P],
                        op=nl.add,
                        axis=1,
                        keepdims=True,
                    )
                else:
                    nisa.memset(dst=ctx.db_part[:co_size, co_t : co_t + 1], value=0.0)
        buf = co_b % ctx.dw_buf_depth
        for cig0 in range(0, ctx.n_ci, _NUM_PSUM_BANKS):
            cig1 = min(cig0 + _NUM_PSUM_BANKS, ctx.n_ci)
            dw_grp = [
                nl.ndarray(
                    shape=(min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile), blk), dtype=_ACC_DTYPE, buffer=nl.psum
                )
                for ci_t in range(cig0, cig1)
            ]
            for si in range(n_slots):
                _, total_rows = slots[si]
                for ci_t in range(cig0, cig1):
                    c0i = ci_t * ctx.ci_tile
                    c1i = min(c0i + ctx.ci_tile, ctx.C_in)
                    nisa.nc_matmul(
                        dst=dw_grp[ci_t - cig0],
                        stationary=xt_res[si][:total_rows, c0i:c1i],
                        moving=dyt_res[si][co_b][:total_rows, :blk],
                        accumulate=(si > 0),
                    )
            for ci_t in range(cig0, cig1):
                ci_size = min(ctx.ci_tile, ctx.C_in - ci_t * ctx.ci_tile)
                src = dw_grp[ci_t - cig0][:ci_size, :blk]
                if ctx.reduce_across:
                    dst = ctx.dw_send[buf][:ci_size, ci_t * ctx.co_blk : ci_t * ctx.co_blk + blk]
                    if ctx.dw_has_work:
                        nisa.tensor_copy(dst=dst, src=src)
                    else:
                        nisa.memset(dst=dst, value=0.0)
                else:
                    nisa.tensor_copy(dst=ctx.dw_sbuf[ci_t][co_b][:ci_size, :blk], src=src)
        if ctx.reduce_across:
            _pw_coblk_reduce_store_pk(ctx, co_b, buf)


def _dwps_alloc(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager, compute_db: bool = False):
    """Allocate all persistent SBUF/PSUM state for phase-split dw."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    dtype = x_in.dtype
    C_in, C_out, K_h, K_w = cfg.C_in, cfg.C_out, cfg.K_h, cfg.K_w
    H, W, H_out, W_out = cfg.H, cfg.W, cfg.H_out, cfg.W_out
    stride_h, stride_w = cfg.stride_h, cfg.stride_w
    pad_h_top, pad_w_left = cfg.pad_h_top, cfg.pad_w_left
    H_pad = H + pad_h_top + cfg.pad_h_bottom
    W_pad = W + pad_w_left + cfg.pad_w_right
    n_cols = cfg.n_cols
    n_pos = H_out * W_out  # per-image output positions (D_out == 1)

    imgs_per_pack = max(1, min(P_MAX // C_in, cfg.B))
    k_full = imgs_per_pack * C_in
    # Number of ho_base row-tiles stacked into one identity-matmul transpose so the
    G_cap = max(1, P_MAX // k_full)

    rows_per_tile = max(1, min(P_MAX // W_out, H_out)) if W_out <= P_MAX else 1
    ptile = rows_per_tile * W_out

    dw = nl.ndarray(shape=(cfg.K_d, K_h, K_w, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)
    db = nl.ndarray(shape=(C_out,), dtype=dy.dtype, buffer=nl.shared_hbm) if compute_db else None
    nc_ext = n_cols + (1 if compute_db else 0)  # extra trailing column holds db
    db_col = n_cols

    identity_buf = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwps_identity")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity_buf)
    ones_buf = sbm.alloc_heap(shape=(P_MAX, 1), dtype=_ACC_DTYPE, name="dwps_ones") if compute_db else None
    if compute_db:
        nisa.memset(dst=ones_buf[:P_MAX, :1], value=1.0)
    _PIPE_DEGREE = cfg.degree("pipe", 3)
    band_h = (rows_per_tile - 1) * stride_h + K_h

    _XPAD_DEGREE = _PIPE_DEGREE
    xpad_bufs = [
        sbm.alloc_heap(shape=(G_cap * k_full, band_h, W_pad), dtype=dtype, name=f"dwps_xpad{i}")
        for i in range(_XPAD_DEGREE)
    ]

    _DY_DEGREE = cfg.degree("dy_t", 2)
    dyT_bufs = [
        sbm.alloc_heap(shape=(ptile, C_out), dtype=dtype, name=f"dwps_dyT{i}", align=32) for i in range(_DY_DEGREE)
    ]

    _XCOL_DEGREE = _PIPE_DEGREE
    xcol_bufs = [
        sbm.alloc_heap(shape=(ptile, K_h * K_w * G_cap * k_full), dtype=dtype, name=f"dwps_xcol{i}")
        for i in range(_XCOL_DEGREE)
    ]

    tp_free = max(C_out, G_cap * k_full)
    psum = _psum_offsets(cfg, "dw_phasesplit")
    (dw_off,) = psum["dw_ps_dw"]
    tp_offsets = psum["dw_ps_tp"]
    _N_TP = len(tp_offsets)
    psum_dw = nl.ndarray(shape=(C_out, n_cols), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, dw_off))
    tp_scratch_banks = [
        nl.ndarray(shape=(P_MAX, tp_free), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off)) for off in tp_offsets
    ]
    tp_scratch = tp_scratch_banks[_N_TP - 1]  # drain-transpose reuses the top bank
    # db rides a dedicated (otherwise idle) tp PSUM bank so its accumulate chain never
    db_psum = tp_scratch_banks[0] if compute_db else None
    # Four fp32 buffers rotate grouped dyT reductions without extending the dependency chain.
    _DB_ACC_N = 4
    gdyT_bufs = (
        [sbm.alloc_heap(shape=(ptile, C_out), dtype=_ACC_DTYPE, name=f"dwps_gdyT{i}") for i in range(_DB_ACC_N)]
        if compute_db
        else None
    )

    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, n_pos]

    x_img = x_in.select(dim=2, index=0)  # [B, C_in, H, W]  (D == 1)
    band_ci_stride = band_h * W_pad  # element distance between (img,ci) rows of a band

    all_ho_bases = list(range(0, H_out, rows_per_tile))
    my_ho_bases = all_ho_bases[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_ho_bases

    return {
        "P_MAX": P_MAX,
        "F_MAX": F_MAX,
        "dtype": dtype,
        "C_in": C_in,
        "C_out": C_out,
        "K_h": K_h,
        "K_w": K_w,
        "H": H,
        "W": W,
        "H_out": H_out,
        "W_out": W_out,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_h_top": pad_h_top,
        "pad_w_left": pad_w_left,
        "W_pad": W_pad,
        "n_cols": n_cols,
        "n_pos": n_pos,
        "imgs_per_pack": imgs_per_pack,
        "k_full": k_full,
        "G_cap": G_cap,
        "rows_per_tile": rows_per_tile,
        "ptile": ptile,
        "dw": dw,
        "db": db,
        "nc_ext": nc_ext,
        "db_col": db_col,
        "identity_buf": identity_buf,
        "ones_buf": ones_buf,
        "_PIPE_DEGREE": _PIPE_DEGREE,
        "band_h": band_h,
        "_XPAD_DEGREE": _XPAD_DEGREE,
        "xpad_bufs": xpad_bufs,
        "_DY_DEGREE": _DY_DEGREE,
        "dyT_bufs": dyT_bufs,
        "_XCOL_DEGREE": _XCOL_DEGREE,
        "xcol_bufs": xcol_bufs,
        "tp_free": tp_free,
        "psum_dw": psum_dw,
        "tp_scratch_banks": tp_scratch_banks,
        "tp_scratch": tp_scratch,
        "db_psum": db_psum,
        "gdyT_bufs": gdyT_bufs,
        "_DB_ACC_N": _DB_ACC_N,
        "dy_flat": dy_flat,
        "x_img": x_img,
        "band_ci_stride": band_ci_stride,
        "all_ho_bases": all_ho_bases,
        "my_ho_bases": my_ho_bases,
    }


def _dwps_plan_groups(st: dict) -> list[list[int]]:
    """Group output-height bands with matching row counts for phase-split processing."""
    rows_per_tile = st["rows_per_tile"]
    G_cap = st["G_cap"]
    H_out = st["H_out"]
    ho_bases = st["my_ho_bases"]
    groups = []
    gi = 0
    while gi < len(ho_bases):
        hb0 = ho_bases[gi]
        rows0 = min(rows_per_tile, H_out - hb0)
        grp = [hb0]
        gi += 1
        while gi < len(ho_bases) and len(grp) < G_cap and min(rows_per_tile, H_out - ho_bases[gi]) == rows0:
            grp.append(ho_bases[gi])
            gi += 1
        groups.append(grp)
    return groups


def _dwps_prefetch(cfg, st):
    """dma-compute-order: physically emit the x-pad DMA loads for pack-0's prologue groups."""
    pack_n = min(st["imgs_per_pack"], cfg.B)
    pack_k = pack_n * st["C_in"]
    groups = _dwps_plan_groups(st)
    _AHEAD = st["_PIPE_DEGREE"] - 1
    xpad_bufs = st["xpad_bufs"]
    _XPAD_DEGREE = st["_XPAD_DEGREE"]
    C_in = st["C_in"]
    band_h = st["band_h"]
    H = st["H"]
    W = st["W"]
    stride_h = st["stride_h"]
    pad_h_top = st["pad_h_top"]
    pad_w_left = st["pad_w_left"]
    x_img = st["x_img"]
    n_pf = min(_AHEAD, len(groups))
    for j in range(n_pf):
        grp = groups[j]
        buf = xpad_bufs[j % _XPAD_DEGREE]
        gk = len(grp) * pack_k
        nisa.memset(dst=buf[:gk, :, :], value=0.0)
        for g, ho_base in enumerate(grp):
            pr0 = ho_base * stride_h
            ih_lo = max(0, pr0 - pad_h_top)
            ih_hi = min(H, pr0 + band_h - pad_h_top)
            n_rows_in = ih_hi - ih_lo
            if n_rows_in <= 0:
                continue
            r0 = ih_lo - (pr0 - pad_h_top)
            for im in range(pack_n):
                xb = x_img.select(dim=0, index=im)
                p0 = g * pack_k + im * C_in
                nisa.dma_copy(
                    dst=buf[p0 : p0 + C_in, r0 : r0 + n_rows_in, pad_w_left : pad_w_left + W],
                    src=xb.slice(dim=1, start=ih_lo, end=ih_hi, step=1),
                )
    return n_pf


def _dwps_compute(
    cfg: Conv3dBwdConfig,
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    sbm: SbufManager,
    st: dict,
    compute_db: bool = False,
    preloaded: int = 0,
):
    """Compute phase-split dw (+ optional db) using operands allocated by _dwps_alloc."""
    P_MAX = st["P_MAX"]
    F_MAX = st["F_MAX"]
    dtype = st["dtype"]
    C_in = st["C_in"]
    C_out = st["C_out"]
    K_h = st["K_h"]
    K_w = st["K_w"]
    H = st["H"]
    W = st["W"]
    H_out = st["H_out"]
    W_out = st["W_out"]
    stride_h = st["stride_h"]
    stride_w = st["stride_w"]
    pad_h_top = st["pad_h_top"]
    pad_w_left = st["pad_w_left"]
    W_pad = st["W_pad"]
    n_cols = st["n_cols"]
    n_pos = st["n_pos"]
    imgs_per_pack = st["imgs_per_pack"]
    k_full = st["k_full"]
    G_cap = st["G_cap"]
    rows_per_tile = st["rows_per_tile"]
    ptile = st["ptile"]
    dw = st["dw"]
    db = st["db"]
    nc_ext = st["nc_ext"]
    db_col = st["db_col"]
    identity_buf = st["identity_buf"]
    ones_buf = st["ones_buf"]
    _PIPE_DEGREE = st["_PIPE_DEGREE"]
    band_h = st["band_h"]
    _XPAD_DEGREE = st["_XPAD_DEGREE"]
    xpad_bufs = st["xpad_bufs"]
    _DY_DEGREE = st["_DY_DEGREE"]
    dyT_bufs = st["dyT_bufs"]
    _XCOL_DEGREE = st["_XCOL_DEGREE"]
    xcol_bufs = st["xcol_bufs"]
    tp_free = st["tp_free"]
    psum_dw = st["psum_dw"]
    tp_scratch_banks = st["tp_scratch_banks"]
    tp_scratch = st["tp_scratch"]
    db_psum = st["db_psum"]
    gdyT_bufs = st["gdyT_bufs"]
    _DB_ACC_N = st["_DB_ACC_N"]
    dy_flat = st["dy_flat"]
    x_img = st["x_img"]
    band_ci_stride = st["band_ci_stride"]
    all_ho_bases = st["all_ho_bases"]

    wrote = False
    db_wrote = False
    for pack_start in range(0, cfg.B, imgs_per_pack):
        pack_n = min(imgs_per_pack, cfg.B - pack_start)

        pack_k = pack_n * C_in
        groups = _dwps_plan_groups(st)

        def _load_group(grp, slot: int) -> None:
            buf = xpad_bufs[slot]
            gk = len(grp) * pack_k
            nisa.memset(dst=buf[:gk, :, :], value=0.0)
            for g, ho_base in enumerate(grp):
                pr0 = ho_base * stride_h  # first padded input row of this band
                ih_lo = max(0, pr0 - pad_h_top)
                ih_hi = min(H, pr0 + band_h - pad_h_top)
                n_rows_in = ih_hi - ih_lo
                if n_rows_in <= 0:
                    continue
                r0 = ih_lo - (pr0 - pad_h_top)  # band-relative row where the sub-band starts
                for im in range(pack_n):
                    xb = x_img.select(dim=0, index=pack_start + im)  # [C_in, H, W]
                    p0 = g * pack_k + im * C_in
                    nisa.dma_copy(
                        dst=buf[p0 : p0 + C_in, r0 : r0 + n_rows_in, pad_w_left : pad_w_left + W],
                        src=xb.slice(dim=1, start=ih_lo, end=ih_hi, step=1),
                    )

        def _build_xcol_group(grp, xpad_slot: int, xcol_slot: int) -> None:
            rows = min(rows_per_tile, H_out - grp[0])
            pt = rows * W_out
            gk = len(grp) * pack_k
            xpad_band = xpad_bufs[xpad_slot]
            xcol = xcol_bufs[xcol_slot]
            for kh in range(K_h):
                for kw in range(K_w):
                    col_base = (kh * K_w + kw) * gk
                    base = kh * W_pad + kw
                    xpad_row = xpad_band.ap(
                        pattern=[[band_ci_stride, gk], [stride_h * W_pad, rows], [stride_w, W_out]],
                        offset=base,
                    )
                    nisa.nc_matmul(
                        dst=tp_scratch[:pt, :gk],
                        stationary=xpad_row,
                        moving=identity_buf[:gk, :gk],
                        accumulate=False,
                    )
                    nisa.tensor_copy(
                        dst=xcol[:pt, col_base : col_base + gk], src=tp_scratch[:pt, :gk], engine=nisa.scalar_engine
                    )

        _AHEAD = _PIPE_DEGREE - 1
        _skip = preloaded if pack_start == 0 else 0
        for j in range(min(_AHEAD, len(groups))):
            if j >= _skip:
                _load_group(groups[j], j % _XPAD_DEGREE)
            _build_xcol_group(groups[j], j % _XPAD_DEGREE, j % _XCOL_DEGREE)
        for gr_idx, grp in enumerate(groups):
            xcol_slot = gr_idx % _XCOL_DEGREE
            xcol_buf = xcol_bufs[xcol_slot]
            rows = min(rows_per_tile, H_out - grp[0])
            pt = rows * W_out
            gk = len(grp) * pack_k

            nxt = gr_idx + _AHEAD
            if nxt < len(groups):
                nxt_xpad = nxt % _XPAD_DEGREE
                _load_group(groups[nxt], nxt_xpad)
                _build_xcol_group(groups[nxt], nxt_xpad, nxt % _XCOL_DEGREE)

            def _dy_transpose(ho_base: int, im: int, slot: int) -> None:
                b = pack_start + im
                ps = ho_base * W_out
                dy_bc = dy_flat.select(dim=0, index=b).slice(dim=1, start=ps, end=ps + pt, step=1)
                nisa.dma_transpose(dst=dyT_bufs[slot][:pt, :C_out], src=dy_bc)

            work = [(g, im) for g in range(len(grp)) for im in range(pack_n)]
            if work:
                g0, im0 = work[0]
                _dy_transpose(grp[g0], im0, 0)  # prime the pipeline
            for w_idx, (g, im) in enumerate(work):
                slot = w_idx % _DY_DEGREE
                if w_idx + 1 < len(work):
                    ng, nim = work[w_idx + 1]
                    _dy_transpose(grp[ng], nim, (w_idx + 1) % _DY_DEGREE)

                xcol_img = xcol_buf.ap(
                    pattern=[[xcol_buf.shape[1], pt], [gk, K_h * K_w], [1, C_in]],
                    offset=g * pack_k + im * C_in,
                )
                nisa.nc_matmul(
                    dst=psum_dw[:C_out, :n_cols],
                    stationary=dyT_bufs[slot][:pt, :C_out],
                    moving=xcol_img,
                    accumulate=wrote,
                )
                if compute_db:
                    # Round-robin across parallel fp32 accumulators to break the serial add chain.
                    a = w_idx % _DB_ACC_N
                    if w_idx < _DB_ACC_N:
                        nisa.tensor_copy(dst=gdyT_bufs[a][:pt, :C_out], src=dyT_bufs[slot][:pt, :C_out])
                    else:
                        nisa.tensor_tensor(
                            dst=gdyT_bufs[a][:pt, :C_out],
                            data1=gdyT_bufs[a][:pt, :C_out],
                            data2=dyT_bufs[slot][:pt, :C_out],
                            op=nl.add,
                        )
                wrote = True
            if compute_db and work:
                # db[co] += sum_p gdyT[p, co] * 1 ; own PSUM bank, own accumulate chain.
                for a in range(min(_DB_ACC_N, len(work))):
                    nisa.nc_matmul(
                        dst=db_psum[:C_out, :1],
                        stationary=gdyT_bufs[a][:pt, :C_out],
                        moving=ones_buf[:pt, :1],
                        accumulate=db_wrote,
                    )
                    db_wrote = True

    result_buf = sbm.alloc_heap(shape=(C_out, n_cols), dtype=dtype, name="dwps_result")
    dw_part = sbm.alloc_heap(shape=(C_out, nc_ext), dtype=_ACC_DTYPE, name="dwps_part")
    if wrote:
        nisa.tensor_copy(dst=dw_part[:C_out, :n_cols], src=psum_dw[:C_out, :n_cols])
        if compute_db:
            nisa.tensor_copy(dst=dw_part[:C_out, db_col : db_col + 1], src=db_psum[:C_out, :1])
    else:
        nisa.memset(dst=dw_part[:C_out, :nc_ext], value=0.0)
    if cfg.is_sharded:
        other_rank = 1 - cfg.prg_id
        dw_recv = sbm.alloc_heap(shape=(C_out, nc_ext), dtype=_ACC_DTYPE, name="dwps_recv")
        nisa.sendrecv(
            src=dw_part[:C_out, :nc_ext],
            dst=dw_recv[:C_out, :nc_ext],
            send_to_rank=other_rank,
            recv_from_rank=other_rank,
            pipe_id=0,
        )
        nisa.tensor_tensor(
            dst=dw_part[:C_out, :nc_ext],
            data1=dw_part[:C_out, :nc_ext],
            data2=dw_recv[:C_out, :nc_ext],
            op=nl.add,
        )
        sbm.pop_heap()  # dw_recv
    if compute_db:
        db_sbuf = sbm.alloc_heap(shape=(C_out, 1), dtype=dy.dtype, name="dwps_db")
        nisa.tensor_copy(dst=db_sbuf[:C_out, :1], src=dw_part[:C_out, db_col : db_col + 1])
        nisa.dma_copy(dst=db, src=db_sbuf[:C_out, 0])
        sbm.pop_heap()  # db_sbuf
    nisa.tensor_copy(dst=result_buf[:C_out, :n_cols], src=dw_part[:C_out, :n_cols])
    sbm.pop_heap()  # dw_part

    dwT_buf = sbm.alloc_heap(shape=(P_MAX, C_out), dtype=dtype, name="dwps_dwT")

    dw_2d = dw.select(dim=0, index=0).reshape((K_h * K_w * C_in, C_out))
    for col_start in range(0, n_cols, P_MAX):
        col_end = min(col_start + P_MAX, n_cols)
        cs = col_end - col_start
        nisa.nc_matmul(
            dst=tp_scratch[:cs, :C_out],
            stationary=result_buf[:C_out, col_start:col_end],
            moving=identity_buf[:C_out, :C_out],
            accumulate=False,
        )
        nisa.tensor_copy(dst=dwT_buf[:cs, :C_out], src=tp_scratch[:cs, :C_out])
        nisa.dma_copy(dst=dw_2d.slice(dim=0, start=col_start, end=col_end, step=1), src=dwT_buf[:cs, :C_out])

    sbm.pop_heap()  # dwT_buf
    sbm.pop_heap()  # result_buf
    for _ in xcol_bufs:
        sbm.pop_heap()  # xcol_bufs
    for _ in dyT_bufs:
        sbm.pop_heap()  # dyT_bufs
    for _ in xpad_bufs:
        sbm.pop_heap()  # xpad_bufs
    if compute_db:
        sbm.pop_heap()  # ones_buf
    sbm.pop_heap()  # identity_buf
    return (dw, db) if compute_db else dw


def _conv3d_dw_phasesplit(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager, compute_db: bool = False
):
    """Efficient dw for the phase-split regime (2D, stride 2, small C_in)."""
    st = _dwps_alloc(cfg, dy, x_in, sbm, compute_db)
    return _dwps_compute(cfg, dy, x_in, sbm, st, compute_db)


def _conv3d_dw_batched(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager) -> nl.NkiTensor:
    """Batched-GEMM dw fallback for any KxK conv (see DwXLoadMethod.BATCHED_GEMM)."""
    P_MAX = nl.tile_size.pmax
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_out = D_out * H_out * W_out
    spatial_in = cfg.D * cfg.H * cfg.W
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w

    co_tile = min(P_MAX, C_out)
    ci_tile = min(P_MAX, C_in)
    n_co = div_ceil(C_out, co_tile)
    n_ci = div_ceil(C_in, ci_tile)

    K_d, K_h, K_w = cfg.K_d, cfg.K_h, cfg.K_w
    taps = [(kd, kh, kw) for kd in range(K_d) for kh in range(K_h) for kw in range(K_w)]
    n_taps = len(taps)

    dw = nl.ndarray(shape=(K_d, K_h, K_w, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)

    if spatial_out >= P_MAX:
        pack = 1
    else:
        pack = min(cfg.B, max(1, div_ceil(4 * P_MAX, spatial_out)))
    L = pack * spatial_out

    dyc = [sbm.alloc_heap(shape=(co_tile, L), dtype=dtype, name=f"dwbg_dyc{i}") for i in range(n_co)]
    x_full = [
        sbm.alloc_heap(shape=(ci_tile, pack * spatial_in), dtype=dtype, name=f"dwbg_xfull{i}") for i in range(n_ci)
    ]
    xc = [sbm.alloc_heap(shape=(ci_tile, L), dtype=dtype, name=f"dwbg_xc{i}") for i in range(n_ci)]

    n_kt = div_ceil(L, P_MAX)
    dy_pos = [
        [sbm.alloc_heap(shape=(P_MAX, co_tile), dtype=dtype, name=f"dwbg_dyp{c}_{kt}", align=32) for kt in range(n_kt)]
        for c in range(n_co)
    ]
    x_pos = [
        [sbm.alloc_heap(shape=(P_MAX, ci_tile), dtype=dtype, name=f"dwbg_xp{c}_{kt}", align=32) for kt in range(n_kt)]
        for c in range(n_ci)
    ]

    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwbg_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)
    result = sbm.alloc_heap(shape=(co_tile, ci_tile), dtype=dtype, name="dwbg_result")

    dw_acc = [
        [
            [
                sbm.alloc_heap(shape=(co_tile, ci_tile), dtype=_ACC_DTYPE, name=f"dwbg_acc{t}_{c}_{i}")
                for i in range(n_ci)
            ]
            for c in range(n_co)
        ]
        for t in range(n_taps)
    ]

    psum = _psum_offsets(cfg, "dw_batched")
    mm_banks = [
        nl.ndarray(shape=(co_tile, ci_tile), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
        for off in psum["dw_bg_mm"]
    ]
    tp_scratch = [
        nl.ndarray(shape=(P_MAX, P_MAX), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off)) for off in psum["dw_bg_tp"]
    ]
    n_tp = len(tp_scratch)
    n_mm = len(mm_banks)

    dy_flat = dy.flatten_dims(2, 4)
    x_flat = x_in.flatten_dims(2, 4)

    packs = [(b0, min(pack, cfg.B - b0)) for b0 in range(0, cfg.B, pack)]

    all_blocks = [(c, i) for c in range(n_co) for i in range(n_ci)]
    my_blocks = [tuple(b) for b in (all_blocks[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_blocks)]
    my_co = sorted({b[0] for b in my_blocks})
    my_ci = sorted({b[1] for b in my_blocks})

    for t in range(n_taps):
        for c in my_co:
            co_len = min(co_tile, C_out - c * co_tile)
            for i in my_ci:
                ci_len = min(ci_tile, C_in - i * ci_tile)
                nisa.memset(dst=dw_acc[t][c][i][:co_len, :ci_len], value=0.0)

    for b0, nb in packs:
        for c in my_co:
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            for j in range(nb):
                nisa.dma_copy(
                    dst=dyc[c][:co_len, j * spatial_out : (j + 1) * spatial_out],
                    src=dy_flat.select(dim=0, index=b0 + j).slice(dim=0, start=co0, end=co0 + co_len, step=1),
                )
        for i in my_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            for j in range(nb):
                nisa.dma_copy(
                    dst=x_full[i][:ci_len, j * spatial_in : (j + 1) * spatial_in],
                    src=x_flat.select(dim=0, index=b0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
                )

        Lp = nb * spatial_out
        kts = [(kt, gp0, min(P_MAX, Lp - gp0)) for kt, gp0 in enumerate(range(0, Lp, P_MAX))]

        tp = 0
        for kt, gp0, klen in kts:
            for c in my_co:
                co_len = min(co_tile, C_out - c * co_tile)
                _pe_transpose(
                    dy_pos[c][kt][:klen, :co_len],
                    dyc[c][:co_len, gp0 : gp0 + klen],
                    tp_scratch[tp],
                    identity,
                    co_len,
                    klen,
                )
                tp = (tp + 1) % n_tp

        for t, (kd, kh, kw) in enumerate(taps):
            do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi = _tap_valid_ranges(cfg, kd, kh, kw)
            empty = do_hi <= do_lo or ho_hi <= ho_lo or wo_hi <= wo_lo
            for i in my_ci:
                ci_len = min(ci_tile, C_in - i * ci_tile)
                nisa.memset(dst=xc[i][:ci_len, :Lp], value=0.0)
                if empty:
                    continue
                xf5 = x_full[i][:ci_len].reshape((ci_len, pack, cfg.D, cfg.H, cfg.W))
                xc5 = xc[i][:ci_len].reshape((ci_len, pack, D_out, H_out, W_out))
                in_d = do_lo * sd + kd - cfg.pad_d_left
                in_h = ho_lo * sh + kh - cfg.pad_h_top
                in_w = wo_lo * sw + kw - cfg.pad_w_left
                src = (
                    xf5.slice(dim=1, start=0, end=nb, step=1)
                    .slice(dim=2, start=in_d, end=in_d + (do_hi - do_lo - 1) * sd + 1, step=sd)
                    .slice(dim=3, start=in_h, end=in_h + (ho_hi - ho_lo - 1) * sh + 1, step=sh)
                    .slice(dim=4, start=in_w, end=in_w + (wo_hi - wo_lo - 1) * sw + 1, step=sw)
                )
                dst = (
                    xc5.slice(dim=1, start=0, end=nb, step=1)
                    .slice(dim=2, start=do_lo, end=do_hi, step=1)
                    .slice(dim=3, start=ho_lo, end=ho_hi, step=1)
                    .slice(dim=4, start=wo_lo, end=wo_hi, step=1)
                )
                nisa.tensor_copy(dst=dst[:ci_len], src=src[:ci_len])

            tp = 0
            for kt, gp0, klen in kts:
                for i in my_ci:
                    ci_len = min(ci_tile, C_in - i * ci_tile)
                    _pe_transpose(
                        x_pos[i][kt][:klen, :ci_len],
                        xc[i][:ci_len, gp0 : gp0 + klen],
                        tp_scratch[tp],
                        identity,
                        ci_len,
                        klen,
                    )
                    tp = (tp + 1) % n_tp

            for blk_idx, (c, i) in enumerate(my_blocks):
                co_len = min(co_tile, C_out - c * co_tile)
                ci_len = min(ci_tile, C_in - i * ci_tile)
                mm_bank = mm_banks[blk_idx % n_mm]
                for kt, gp0, klen in kts:
                    nisa.nc_matmul(
                        dst=mm_bank[:co_len, :ci_len],
                        stationary=dy_pos[c][kt][:klen, :co_len],
                        moving=x_pos[i][kt][:klen, :ci_len],
                        accumulate=(kt > 0),
                    )
                acc = dw_acc[t][c][i]
                nisa.tensor_tensor(
                    dst=acc[:co_len, :ci_len], data1=acc[:co_len, :ci_len], data2=mm_bank[:co_len, :ci_len], op=nl.add
                )

    for t, (kd, kh, kw) in enumerate(taps):
        dw_tap = dw.select(dim=0, index=kd).select(dim=0, index=kh).select(dim=0, index=kw)  # [C_in, C_out]
        for c, i in my_blocks:
            co0 = c * co_tile
            ci0 = i * ci_tile
            co_len = min(co_tile, C_out - co0)
            ci_len = min(ci_tile, C_in - ci0)
            nisa.tensor_copy(dst=result[:co_len, :ci_len], src=dw_acc[t][c][i][:co_len, :ci_len])
            dst = dw_tap.ap(pattern=[[1, co_len], [C_out, ci_len]], offset=ci0 * C_out + co0)
            nisa.dma_copy(dst=dst, src=result[:co_len, :ci_len])

    for _ in range(n_taps):
        for _ in range(n_co):
            for _ in range(n_ci):
                sbm.pop_heap()  # dw_acc
    sbm.pop_heap()  # result
    sbm.pop_heap()  # identity
    for _ in range(n_ci):
        for _ in range(n_kt):
            sbm.pop_heap()  # x_pos
    for _ in range(n_co):
        for _ in range(n_kt):
            sbm.pop_heap()  # dy_pos
    for _ in range(n_ci):
        sbm.pop_heap()  # xc
    for _ in range(n_ci):
        sbm.pop_heap()  # x_full
    for _ in range(n_co):
        sbm.pop_heap()  # dyc
    return dw


def _conv3d_dw_pointwise_strided(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager
) -> nl.NkiTensor:
    """1x1 pointwise dw as a plain GEMM (see DwXLoadMethod.POINTWISE)."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_out = D_out * H_out * W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w
    strided = sd != 1 or sh != 1 or sw != 1

    ci_tile = min(P_MAX, C_in)
    co_blk = min(F_MAX, C_out)
    n_ci = div_ceil(C_in, ci_tile)
    n_cb = div_ceil(C_out, co_blk)

    dw = nl.ndarray(shape=(1, 1, 1, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)
    dw_flat = dw.reshape((C_in, C_out))

    _PP = cfg.degree("operands", 2)
    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwp_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)
    _NTP = 4
    # Transpose scratch sits in the top banks; accumulators claim banks [0, _MAX_ACC_BANKS).
    tp_scratch = [
        nl.ndarray(
            shape=(P_MAX, P_MAX),
            dtype=_ACC_DTYPE,
            buffer=nl.psum,
            address=(0, (_NUM_PSUM_BANKS - _NTP + b) * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
        )
        for b in range(_NTP)
    ]
    _MAX_ACC_BANKS = _NUM_PSUM_BANKS - _NTP

    xt = [
        [sbm.alloc_heap(shape=(P_MAX, ci_tile), dtype=dtype, name=f"dwp_xt{p}_{i}", align=32) for i in range(n_ci)]
        for p in range(_PP)
    ]
    spatial_in = cfg.D * cfg.H * cfg.W
    G = cfg.B * spatial_out
    # Pack positions across batches when the output plane is smaller than the PE.
    pack = (spatial_out < P_MAX) and (cfg.B > 1)
    # x staging (channel-major) is needed for strided subsampling and for packed staging.
    stage_x = strided or pack
    xstage_w = G if pack else spatial_out
    xplane_w = (cfg.B * spatial_in) if pack else spatial_in
    n_csub = div_ceil(co_blk, P_MAX)

    # Block assignment for this core (strided across the (ci, cb) grid when sharded).
    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_cb)]
    my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_blocks
    block_groups = [my_blocks[g : g + _MAX_ACC_BANKS] for g in range(0, len(my_blocks), _MAX_ACC_BANKS)]
    my_cb = sorted({c for _, c in my_blocks})
    ncb_local = len(my_cb)
    cb_slot = {c: k for k, c in enumerate(my_cb)}
    n_steps = div_ceil(G, P_MAX) if pack else cfg.B * div_ceil(spatial_out, P_MAX)
    # Hoist dy staging + transpose out of the block-group loop (shared across block-groups).
    hoist_dy = pack

    x_plane = (
        [sbm.alloc_heap(shape=(ci_tile, xplane_w), dtype=dtype, name=f"dwp_xplane{i}") for i in range(n_ci)]
        if stage_x
        else None
    )
    x_stage = (
        [sbm.alloc_heap(shape=(ci_tile, xstage_w), dtype=dtype, name=f"dwp_xstage{i}") for i in range(n_ci)]
        if stage_x
        else None
    )
    # Channel-major dy staging (positions packed across batches in the free dim).
    dy_pack = (
        [
            [sbm.alloc_heap(shape=(P_MAX, G), dtype=dtype, name=f"dwp_dypack{k}_{s}") for s in range(n_csub)]
            for k in range(ncb_local)
        ]
        if pack
        else None
    )
    if hoist_dy:
        # Resident transposed dy for all steps (shared across block-groups).
        dyt_all = [
            [
                sbm.alloc_heap(shape=(P_MAX, co_blk), dtype=dtype, name=f"dwp_dyt{si}_{k}", align=32)
                for k in range(ncb_local)
            ]
            for si in range(n_steps)
        ]
        dyt = None
    else:
        dyt = [
            [
                sbm.alloc_heap(shape=(P_MAX, co_blk), dtype=dtype, name=f"dwp_dyt{p}_{k}", align=32)
                for k in range(ncb_local)
            ]
            for p in range(_PP)
        ]
        dyt_all = None
    result = sbm.alloc_heap(shape=(ci_tile, co_blk), dtype=dtype, name="dwp_result")

    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, spatial_out]
    x_flat = x_in.flatten_dims(2, 4)  # [B, C_in, spatial_in]

    def _stage_x(b, grp_ci, col_off):
        for i in grp_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            nisa.dma_copy(
                dst=x_plane[i][:ci_len, :],
                src=x_flat.select(dim=0, index=b).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
            )
            grid = (
                x_plane[i][:ci_len]
                .reshape((ci_len, cfg.D, cfg.H, cfg.W))
                .slice(dim=1, start=0, end=D_out * sd, step=sd)
                .slice(dim=2, start=0, end=H_out * sh, step=sh)
                .slice(dim=3, start=0, end=W_out * sw, step=sw)
            )
            nisa.tensor_copy(
                dst=x_stage[i][:ci_len, col_off : col_off + spatial_out].reshape((ci_len, D_out, H_out, W_out)),
                src=grid,
            )

    def _stage_x_all(grp_ci):
        # Coalesced staging: load all batches of x in one strided DMA per ci tile.
        for i in grp_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            nisa.dma_copy(
                dst=x_plane[i][:ci_len, :],
                src=x_flat.ap(
                    pattern=[[spatial_in, ci_len], [C_in * spatial_in, cfg.B], [1, spatial_in]],
                    offset=ci0 * spatial_in,
                ),
            )
            grid = (
                x_plane[i][:ci_len]
                .reshape((ci_len, cfg.B, cfg.D, cfg.H, cfg.W))
                .slice(dim=2, start=0, end=D_out * sd, step=sd)
                .slice(dim=3, start=0, end=H_out * sh, step=sh)
                .slice(dim=4, start=0, end=W_out * sw, step=sw)
            )
            nisa.tensor_copy(
                dst=x_stage[i][:ci_len].reshape((ci_len, cfg.B, D_out, H_out, W_out)),
                src=grid,
            )

    def _stage_dy(grp_cb):
        # Coalesced channel-major dy staging: one strided DMA per (co block, sub).
        for c in grp_cb:
            co0 = c * co_blk
            cb_len = min(co_blk, C_out - co0)
            for s in range(n_csub):
                cs0 = s * P_MAX
                cs_len = min(P_MAX, cb_len - cs0)
                if cs_len <= 0:
                    continue
                nisa.dma_copy(
                    dst=dy_pack[cb_slot[c]][s][:cs_len, :G],
                    src=dy_flat.ap(
                        pattern=[[spatial_out, cs_len], [C_out * spatial_out, cfg.B], [1, spatial_out]],
                        offset=(co0 + cs0) * spatial_out,
                    ),
                )

    def _transpose(step, slot, grp_ci, grp_cb, do_dy=True):
        g0, g_len, segs = step
        if pack:
            # x_stage / dy_pack pack positions contiguously in the free dim, so a
            for i in grp_ci:
                ci0 = i * ci_tile
                ci_len = min(ci_tile, C_in - ci0)
                nisa.dma_transpose(dst=xt[slot][i][:g_len, :ci_len], src=x_stage[i][:ci_len, g0 : g0 + g_len])
            if do_dy:
                for c in grp_cb:
                    cb_len = min(co_blk, C_out - c * co_blk)
                    for s in range(n_csub):
                        cs0 = s * P_MAX
                        cs_len = min(P_MAX, cb_len - cs0)
                        if cs_len <= 0:
                            continue
                        nisa.dma_transpose(
                            dst=dyt[slot][cb_slot[c]][:g_len, cs0 : cs0 + cs_len],
                            src=dy_pack[cb_slot[c]][s][:cs_len, g0 : g0 + g_len],
                        )
            return
        for i in grp_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            if strided:
                src = x_stage[i][:ci_len, g0 : g0 + g_len]
            else:
                b, pos0, seg_len, row = segs[0]
                src = (
                    x_flat.select(dim=0, index=b)
                    .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
                    .slice(dim=1, start=pos0, end=pos0 + seg_len, step=1)
                )
            nisa.dma_transpose(dst=xt[slot][i][:g_len, :ci_len], src=src)
        b, pos0, seg_len, row = segs[0]
        if do_dy:
            for c in grp_cb:
                co0 = c * co_blk
                cb_len = min(co_blk, C_out - co0)
                src = (
                    dy_flat.select(dim=0, index=b)
                    .slice(dim=0, start=co0, end=co0 + cb_len, step=1)
                    .slice(dim=1, start=pos0, end=pos0 + seg_len, step=1)
                )
                nisa.dma_transpose(dst=dyt[slot][cb_slot[c]][:g_len, :cb_len], src=src)

    # Build global (batch, position) contraction tiles, each covering up to P_MAX positions.
    steps = []
    if pack:
        for g0 in range(0, G, P_MAX):
            g_len = min(P_MAX, G - g0)
            segs = []
            row = 0
            p = g0
            while p < g0 + g_len:
                b = p // spatial_out
                pos_in_b = p % spatial_out
                seg_len = min(spatial_out - pos_in_b, g0 + g_len - p)
                segs.append((b, pos_in_b, seg_len, row))
                row += seg_len
                p += seg_len
            steps.append((g0, g_len, segs))
    else:
        for b in range(cfg.B):
            for pos0 in range(0, spatial_out, P_MAX):
                pl = min(P_MAX, spatial_out - pos0)
                steps.append((b * spatial_out + pos0, pl, [(b, pos0, pl, 0)]))
    if hoist_dy:
        # Stage + transpose dy once for this core's C_out blocks across all steps.
        _stage_dy(my_cb)
        _tpi = 0
        for si, step in enumerate(steps):
            g0, g_len, segs = step
            for c in my_cb:
                cb_len = min(co_blk, C_out - c * co_blk)
                for s in range(n_csub):
                    cs0 = s * P_MAX
                    cs_len = min(P_MAX, cb_len - cs0)
                    if cs_len <= 0:
                        continue
                    _pe_transpose(
                        dyt_all[si][cb_slot[c]][:g_len, cs0 : cs0 + cs_len],
                        dy_pack[cb_slot[c]][s][:cs_len, g0 : g0 + g_len],
                        tp_scratch[_tpi % _NTP],
                        identity,
                        cs_len,
                        g_len,
                    )
                    _tpi += 1
    for gi, grp in enumerate(block_groups):
        grp_ci = sorted({i for i, _ in grp})
        grp_cb = sorted({c for _, c in grp})
        psum_banks = {
            (i, c): nl.ndarray(
                shape=(ci_tile, co_blk),
                dtype=_ACC_DTYPE,
                buffer=nl.psum,
                address=(0, bi * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
            )
            for bi, (i, c) in enumerate(grp)
        }
        if pack:
            _stage_x_all(grp_ci)
            if not hoist_dy:
                _stage_dy(grp_cb)
            # Hoist the x position-major transpose onto the PE as well.
            ci_local = {i: k for k, i in enumerate(grp_ci)}
            xt_grp = [
                [
                    sbm.alloc_heap(shape=(P_MAX, ci_tile), dtype=dtype, name=f"dwp_xtg{gi}_{si}_{k}", align=32)
                    for k in range(len(grp_ci))
                ]
                for si in range(n_steps)
            ]
            tpx = 0
            for si, step in enumerate(steps):
                g0, g_len, segs = step
                for i in grp_ci:
                    ci_len = min(ci_tile, C_in - i * ci_tile)
                    _pe_transpose(
                        xt_grp[si][ci_local[i]][:g_len, :ci_len],
                        x_stage[i][:ci_len, g0 : g0 + g_len],
                        tp_scratch[tpx % _NTP],
                        identity,
                        ci_len,
                        g_len,
                    )
                    tpx += 1
            for si, step in enumerate(steps):
                g_len = step[1]
                for i, c in grp:
                    ci_len = min(ci_tile, C_in - i * ci_tile)
                    cb_len = min(co_blk, C_out - c * co_blk)
                    nisa.nc_matmul(
                        dst=psum_banks[(i, c)][:ci_len, :cb_len],
                        stationary=xt_grp[si][ci_local[i]][:g_len, :ci_len],
                        moving=dyt_all[si][cb_slot[c]][:g_len, :cb_len],
                        accumulate=(si > 0),
                    )
            for _ in range(n_steps):
                for _ in range(len(grp_ci)):
                    sbm.pop_heap()  # xt_grp
            for i, c in grp:
                ci0 = i * ci_tile
                co0 = c * co_blk
                ci_len = min(ci_tile, C_in - ci0)
                cb_len = min(co_blk, C_out - co0)
                nisa.tensor_copy(dst=result[:ci_len, :cb_len], src=psum_banks[(i, c)][:ci_len, :cb_len])
                nisa.dma_copy(
                    dst=dw_flat.ap(pattern=[[C_out, ci_len], [1, cb_len]], offset=ci0 * C_out + co0),
                    src=result[:ci_len, :cb_len],
                )
            continue
        if strided and steps:
            _stage_x(steps[0][2][0][0], grp_ci, 0)
        staged_b = steps[0][2][0][0] if (steps and not pack) else -1
        if steps:
            _transpose(steps[0], 0, grp_ci, grp_cb, do_dy=not hoist_dy)
        for si, step in enumerate(steps):
            slot = si % _PP
            g_len = step[1]
            if si + 1 < len(steps):
                if strided and not pack:
                    nb = steps[si + 1][2][0][0]
                    if nb != staged_b:
                        _stage_x(nb, grp_ci, 0)
                        staged_b = nb
                _transpose(steps[si + 1], (si + 1) % _PP, grp_ci, grp_cb, do_dy=not hoist_dy)
            for i, c in grp:
                ci_len = min(ci_tile, C_in - i * ci_tile)
                cb_len = min(co_blk, C_out - c * co_blk)
                mov = dyt_all[si][cb_slot[c]] if hoist_dy else dyt[slot][cb_slot[c]]
                nisa.nc_matmul(
                    dst=psum_banks[(i, c)][:ci_len, :cb_len],
                    stationary=xt[slot][i][:g_len, :ci_len],
                    moving=mov[:g_len, :cb_len],
                    accumulate=(si > 0),
                )
        for i, c in grp:
            ci0 = i * ci_tile
            co0 = c * co_blk
            ci_len = min(ci_tile, C_in - ci0)
            cb_len = min(co_blk, C_out - co0)
            nisa.tensor_copy(dst=result[:ci_len, :cb_len], src=psum_banks[(i, c)][:ci_len, :cb_len])
            nisa.dma_copy(
                dst=dw_flat.ap(pattern=[[C_out, ci_len], [1, cb_len]], offset=ci0 * C_out + co0),
                src=result[:ci_len, :cb_len],
            )

    sbm.pop_heap()  # result
    if hoist_dy:
        for _ in range(n_steps):
            for _ in range(ncb_local):
                sbm.pop_heap()  # dyt_all
    else:
        for _ in range(_PP):
            for _ in range(ncb_local):
                sbm.pop_heap()  # dyt
    if dy_pack != None:
        for c in dy_pack:
            for _ in c:
                sbm.pop_heap()  # dy_pack
    if x_stage != None:
        for _ in x_stage:
            sbm.pop_heap()
    if x_plane != None:
        for _ in x_plane:
            sbm.pop_heap()
    for _ in range(_PP):
        for _ in range(n_ci):
            sbm.pop_heap()  # xt
    sbm.pop_heap()  # identity
    return dw


def _conv3d_dw_pointwise(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager) -> nl.NkiTensor:
    """1x1 pointwise dw as a plain GEMM (see DwXLoadMethod.POINTWISE)."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_out = D_out * H_out * W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w
    strided = sd != 1 or sh != 1 or sw != 1

    ci_tile = min(P_MAX, C_in)
    co_blk = min(F_MAX, C_out)
    n_ci = div_ceil(C_in, ci_tile)
    n_cb = div_ceil(C_out, co_blk)

    dw = nl.ndarray(shape=(1, 1, 1, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)
    dw_flat = dw.reshape((C_in, C_out))

    _PP = cfg.degree("operands", 2)
    xt = [
        [sbm.alloc_heap(shape=(P_MAX, ci_tile), dtype=dtype, name=f"dwp_xt{p}_{i}", align=32) for i in range(n_ci)]
        for p in range(_PP)
    ]
    spatial_in = cfg.D * cfg.H * cfg.W
    x_plane = (
        [sbm.alloc_heap(shape=(ci_tile, spatial_in), dtype=dtype, name=f"dwp_xplane{i}") for i in range(n_ci)]
        if strided
        else None
    )
    x_stage = (
        [sbm.alloc_heap(shape=(ci_tile, spatial_out), dtype=dtype, name=f"dwp_xstage{i}") for i in range(n_ci)]
        if strided
        else None
    )
    dyt = [
        [sbm.alloc_heap(shape=(P_MAX, co_blk), dtype=dtype, name=f"dwp_dyt{p}_{c}", align=32) for c in range(n_cb)]
        for p in range(_PP)
    ]
    result = sbm.alloc_heap(shape=(ci_tile, co_blk), dtype=dtype, name="dwp_result")

    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, spatial_out]
    x_flat = x_in.flatten_dims(2, 4)  # [B, C_in, spatial_in]

    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_cb)]
    my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_blocks

    block_groups = [my_blocks[g : g + _NUM_PSUM_BANKS] for g in range(0, len(my_blocks), _NUM_PSUM_BANKS)]

    def _stage_x(b, grp_ci):
        for i in grp_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            nisa.dma_copy(
                dst=x_plane[i][:ci_len, :],
                src=x_flat.select(dim=0, index=b).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
            )
            grid = (
                x_plane[i][:ci_len]
                .reshape((ci_len, cfg.D, cfg.H, cfg.W))
                .slice(dim=1, start=0, end=D_out * sd, step=sd)
                .slice(dim=2, start=0, end=H_out * sh, step=sh)
                .slice(dim=3, start=0, end=W_out * sw, step=sw)
            )
            nisa.tensor_copy(dst=x_stage[i][:ci_len].reshape((ci_len, D_out, H_out, W_out)), src=grid)

    def _transpose(step, slot, grp_ci, grp_cb):
        b, pos0, pos_len = step
        for i in grp_ci:
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            if strided:
                src = x_stage[i][:ci_len, pos0 : pos0 + pos_len]
            else:
                src = (
                    x_flat.select(dim=0, index=b)
                    .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
                    .slice(dim=1, start=pos0, end=pos0 + pos_len, step=1)
                )
            nisa.dma_transpose(dst=xt[slot][i][:pos_len, :ci_len], src=src)
        for c in grp_cb:
            co0 = c * co_blk
            cb_len = min(co_blk, C_out - co0)
            src = (
                dy_flat.select(dim=0, index=b)
                .slice(dim=0, start=co0, end=co0 + cb_len, step=1)
                .slice(dim=1, start=pos0, end=pos0 + pos_len, step=1)
            )
            nisa.dma_transpose(dst=dyt[slot][c][:pos_len, :cb_len], src=src)

    steps = [(b, pos0, min(P_MAX, spatial_out - pos0)) for b in range(cfg.B) for pos0 in range(0, spatial_out, P_MAX)]
    for grp in block_groups:
        grp_ci = sorted({i for i, _ in grp})
        grp_cb = sorted({c for _, c in grp})
        psum_banks = {
            (i, c): nl.ndarray(
                shape=(ci_tile, co_blk),
                dtype=_ACC_DTYPE,
                buffer=nl.psum,
                address=(0, bi * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
            )
            for bi, (i, c) in enumerate(grp)
        }
        staged_b = -1
        if strided and steps:
            _stage_x(steps[0][0], grp_ci)
            staged_b = steps[0][0]
        if steps:
            _transpose(steps[0], 0, grp_ci, grp_cb)
        for si, step in enumerate(steps):
            slot = si % _PP
            b, pos0, pos_len = step
            if si + 1 < len(steps):
                nb = steps[si + 1][0]
                if strided and nb != staged_b:
                    _stage_x(nb, grp_ci)
                    staged_b = nb
                _transpose(steps[si + 1], (si + 1) % _PP, grp_ci, grp_cb)
            for i, c in grp:
                ci_len = min(ci_tile, C_in - i * ci_tile)
                cb_len = min(co_blk, C_out - c * co_blk)
                nisa.nc_matmul(
                    dst=psum_banks[(i, c)][:ci_len, :cb_len],
                    stationary=xt[slot][i][:pos_len, :ci_len],
                    moving=dyt[slot][c][:pos_len, :cb_len],
                    accumulate=(si > 0),
                )
        for i, c in grp:
            ci0 = i * ci_tile
            co0 = c * co_blk
            ci_len = min(ci_tile, C_in - ci0)
            cb_len = min(co_blk, C_out - co0)
            nisa.tensor_copy(dst=result[:ci_len, :cb_len], src=psum_banks[(i, c)][:ci_len, :cb_len])
            nisa.dma_copy(
                dst=dw_flat.ap(pattern=[[C_out, ci_len], [1, cb_len]], offset=ci0 * C_out + co0),
                src=result[:ci_len, :cb_len],
            )

    sbm.pop_heap()  # result
    for _ in range(_PP):
        for _ in range(n_cb):
            sbm.pop_heap()  # dyt
    if x_stage != None:
        for _ in x_stage:
            sbm.pop_heap()
    if x_plane != None:
        for _ in x_plane:
            sbm.pop_heap()
    for _ in range(_PP):
        for _ in range(n_ci):
            sbm.pop_heap()  # xt
    return dw


def _conv3d_dw_fast(
    cfg: Conv3dBwdConfig,
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    sbm: SbufManager,
    want_db: bool = False,
    prefetch: dict | None = None,
):
    """Resident-plane dw for multi-tap / true-3D (FAST_PLANE): transpose each pack's dy/xs and GEMM."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    # Cap each vector copy tile at the vector engine's 2048-element free-dimension limit.
    VECTOR_COPY_F_MAX = 2048
    # Use vector for every third transpose eviction to leave capacity for xs staging.
    VECTOR_COPY_INTERVAL = 3
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D, H, W = cfg.D, cfg.H, cfg.W
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_in = D * H * W
    spatial_out = D_out * H_out * W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w

    ci_tile = min(P_MAX, C_in)
    co_tile = min(P_MAX, C_out)
    n_ci = div_ceil(C_in, ci_tile)
    n_co = div_ceil(C_out, co_tile)

    taps = []
    for kd in range(cfg.K_d):
        do_lo, do_hi = _valid_output_range(D_out, D, kd, sd, 1, cfg.pad_d_left)
        for kh in range(cfg.K_h):
            ho_lo, ho_hi = _valid_output_range(H_out, H, kh, sh, 1, cfg.pad_h_top)
            for kw in range(cfg.K_w):
                wo_lo, wo_hi = _valid_output_range(W_out, W, kw, sw, 1, cfg.pad_w_left)
                taps.append((kd, kh, kw, do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi))
    n_taps = len(taps)

    dw = nl.ndarray(shape=(cfg.K_d, cfg.K_h, cfg.K_w, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)

    grid_can_shard = (n_ci * n_co) >= cfg.n_prgs and (n_ci * n_co) > 1
    batch_shard = cfg.is_sharded and not grid_can_shard
    # db is folded into this pass' dy loads only when every core sees the full batch.
    do_db = want_db and not batch_shard
    db = nl.ndarray(shape=(C_out,), dtype=dtype, buffer=nl.shared_hbm) if do_db else None
    if batch_shard:
        b_per = div_ceil(cfg.B, cfg.n_prgs)
        b_lo = min(b_per * cfg.prg_id, cfg.B)
        b_hi = min(b_lo + b_per, cfg.B)
    else:
        b_lo, b_hi = 0, cfg.B
    B_local = b_hi - b_lo

    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_co)]
    ci_sharded = False
    if grid_can_shard and not batch_shard and n_ci >= cfg.n_prgs:
        ci_sharded = True
        # Shard on C_in tiles: each core owns a disjoint set of C_in tiles.
        my_ci = list(range(n_ci))[cfg.prg_id :: cfg.n_prgs]
        my_blocks = [(i, c) for i in my_ci for c in range(n_co)]
    elif grid_can_shard:
        my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs]
    else:
        my_blocks = all_blocks

    ds = sizeinbytes(dtype)
    if prefetch != None:
        identity = prefetch["identity"]
        fast_pack = prefetch["fast_pack"]
        fast_L = prefetch["fast_L"]
        n_pos_fast = prefetch["n_pos_fast"]
    else:
        identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwf_id")
        nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)
        per_batch_bytes = (spatial_in + n_co * spatial_out + n_co * co_tile + 2 * spatial_out + 2 * ci_tile) * ds
        free_room = max(0, (sbm.get_free_space() - _SBUF_HEADROOM) // 2)
        fast_pack = max(1, min(B_local, free_room // max(1, per_batch_bytes)))
        fast_L = fast_pack * spatial_out
        n_pos_fast = div_ceil(fast_L, P_MAX)

    # Double-buffer input planes so DMA loads can overlap compute on the preceding channel tile.
    x_planes = [
        sbm.alloc_heap(shape=(ci_tile, fast_pack * spatial_in), dtype=dtype, name=f"dwf_xplane{b}") for b in range(2)
    ]
    if prefetch != None:
        y_plane = prefetch["y_plane"]
    else:
        y_plane = [sbm.alloc_heap(shape=(co_tile, fast_L), dtype=dtype, name=f"dwf_yplane{c}") for c in range(n_co)]
    # yt: single contiguous positions-major dy buffer laid out (n_pos_fast, C_out_total)
    co_total = n_co * co_tile
    yt = sbm.alloc_heap(shape=(P_MAX, n_pos_fast * co_total), dtype=dtype, name="dwf_yt", align=32)
    yt3 = yt[:, : n_pos_fast * co_total].reshape((P_MAX, n_pos_fast, co_total))
    # Double-buffer the xs (positions-major staging) and xt (transposed) scratch tiles
    xs = [sbm.alloc_heap(shape=(ci_tile, fast_L), dtype=dtype, name=f"dwf_xs{b}") for b in range(2)]
    xt = [
        sbm.alloc_heap(shape=(P_MAX, n_pos_fast * ci_tile), dtype=dtype, name=f"dwf_xt{b}", align=32) for b in range(2)
    ]
    # C_in tiles this core owns, and the C_out tiles per C_in tile.
    ci_ts = sorted({b[0] for b in my_blocks})
    co_by_ci = {t: [c for (i, c) in my_blocks if i == t] for t in ci_ts}
    all_co = sorted({c for cos in co_by_ci.values() for c in cos})
    # db owner: each C_out block's bias-gradient reduction is done by exactly one core.
    if ci_sharded:
        db_co = set(all_co[cfg.prg_id :: cfg.n_prgs])
    else:
        db_co = set(all_co)
    n_ci_local = len(ci_ts)
    dw_acc = [
        sbm.alloc_heap(shape=(ci_tile, n_taps, n_co, co_tile), dtype=_ACC_DTYPE, name=f"dwf_acc{k}")
        for k in range(n_ci_local)
    ]
    dw_recv = (
        [
            sbm.alloc_heap(shape=(ci_tile, n_taps, n_co, co_tile), dtype=_ACC_DTYPE, name=f"dwf_recv{k}")
            for k in range(n_ci_local)
        ]
        if batch_shard
        else None
    )
    result = [sbm.alloc_heap(shape=(ci_tile, co_total), dtype=dtype, name=f"dwf_result{r}") for r in range(2)]
    db_acc = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dwf_dbacc") if do_db else None
    db_partial = sbm.alloc_heap(shape=(co_tile, 1), dtype=_ACC_DTYPE, name="dwf_dbpart") if do_db else None
    if do_db:
        nisa.memset(dst=db_acc[:co_tile, :], value=0.0)

    psum = _psum_offsets(cfg, "dw_fast")
    # acc spans up to a full PSUM bank (512 fp32) so the dw GEMM can target the whole co block.
    acc_width = min(_PSUM_BANK_FMAX, n_co * co_tile)
    acc_banks = [
        nl.ndarray(shape=(ci_tile, acc_width), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
        for off in psum["dw_fast_acc"]
    ]
    transpose_f_max = F_MAX * _ACC_DTYPE_SIZE // ds
    scratch = [
        nl.ndarray(shape=(P_MAX, transpose_f_max), dtype=dtype, buffer=nl.psum, address=(0, off))
        for off in psum["dw_fast_tp"]
    ]
    n_scr = len(scratch)

    x_flat = x_in.flatten_dims(2, 4)
    y_flat = dy.flatten_dims(2, 4)

    # Skip the full-tensor memset of dw_acc: the first batch pack folds via copy (only empty taps memset).
    if B_local == 0:
        skipped_taps = list(range(n_taps))
    else:
        skipped_taps = [
            ti
            for ti, (_kd, _kh, _kw, do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi) in enumerate(taps)
            if do_hi <= do_lo or ho_hi <= ho_lo or wo_hi <= wo_lo
        ]
    for k in range(n_ci_local):
        ci0 = ci_ts[k] * ci_tile
        ci_len = min(ci_tile, C_in - ci0)
        for ti in skipped_taps:
            nisa.memset(dst=dw_acc[k][:ci_len, ti, :, :], value=0.0)

    tp = 0
    acc_bank_idx = 0
    x_plane_idx = 0
    for b0 in range(b_lo, b_hi, fast_pack):
        nb = min(fast_pack, b_hi - b0)
        pack_pos = nb * spatial_out
        n_pt = div_ceil(pack_pos, P_MAX)

        # Build positions-major dy (yt) once per pack for every owned C_out tile.
        for c in all_co:
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            y_dst = y_plane[c][:co_len].reshape((co_len, fast_pack, spatial_out)).slice(dim=1, start=0, end=nb, step=1)
            if not (prefetch != None and b0 == b_lo):
                nisa.dma_copy(
                    dst=y_dst,
                    src=y_flat.ap(
                        pattern=[[spatial_out, co_len], [C_out * spatial_out, nb], [1, spatial_out]],
                        offset=b0 * C_out * spatial_out + co0 * spatial_out,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )
            if do_db and c in db_co:
                # Fold bias-gradient reduction: db[co] += sum over this pack's positions.
                pack_cols = nb * spatial_out
                nisa.tensor_reduce(
                    dst=db_partial[:co_len, :], op=nl.add, data=y_plane[c][:co_len, :pack_cols], axis=1, keepdims=True
                )
                nisa.tensor_tensor(
                    dst=db_acc[:co_len, c : c + 1],
                    data1=db_acc[:co_len, c : c + 1],
                    data2=db_partial[:co_len, :],
                    op=nl.add,
                )
            yt_tile = yt3[:, :, co0 : co0 + co_len]
            g_co = max(1, transpose_f_max // co_len)
            t = 0
            while t < n_pt:
                g = min(g_co, n_pt - t)
                sc = scratch[tp][:, : g * co_len].reshape((P_MAX, g, co_len))
                drain_p = min(P_MAX, pack_pos - t * P_MAX)
                for j in range(g):
                    p0 = (t + j) * P_MAX
                    ptile = min(P_MAX, pack_pos - p0)
                    nisa.nc_matmul(
                        dst=sc[:ptile, j, :co_len],
                        stationary=y_plane[c][:co_len, p0 : p0 + ptile],
                        moving=identity[:co_len, :co_len],
                        accumulate=False,
                        is_transpose=True,
                    )
                nisa.tensor_copy(
                    dst=yt_tile[:drain_p, t : t + g, :co_len],
                    src=sc[:drain_p, :g, :co_len],
                    engine=nisa.scalar_engine if tp % 2 == 0 else nisa.vector_engine,
                )
                tp = (tp + 1) % n_scr
                t += g
        copy_balance_count = 0
        for k in range(n_ci_local):
            ci_t = ci_ts[k]
            ci0 = ci_t * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            my_co = co_by_ci[ci_t]
            g_ci = max(1, transpose_f_max // ci_len)
            cur_x_plane = x_planes[x_plane_idx]
            x_plane_idx ^= 1

            for j in range(nb):
                nisa.dma_copy(
                    dst=cur_x_plane[:ci_len, j * spatial_in : (j + 1) * spatial_in],
                    src=x_flat.select(dim=0, index=b0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
                    dge_mode=nisa.dge_mode.none,
                )
            xplane5 = cur_x_plane[:ci_len].reshape((ci_len, fast_pack, D, H, W))

            xb = 0
            for tap_idx, (kd, kh, kw, do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi) in enumerate(taps):
                if do_hi <= do_lo or ho_hi <= ho_lo or wo_hi <= wo_lo:
                    continue
                cur_xs = xs[xb]  # alternate scratch per tap so the next tap can overlap
                cur_xt = xt[xb]
                xb ^= 1
                xs5 = cur_xs[:ci_len].reshape((ci_len, fast_pack, D_out, H_out, W_out))
                xsp = xs5.slice(dim=1, start=0, end=nb, step=1)
                if do_lo > 0:
                    nisa.memset(dst=xsp.slice(dim=2, start=0, end=do_lo, step=1), value=0.0)
                if do_hi < D_out:
                    nisa.memset(dst=xsp.slice(dim=2, start=do_hi, end=D_out, step=1), value=0.0)
                mid_d = xsp.slice(dim=2, start=do_lo, end=do_hi, step=1)
                if ho_lo > 0:
                    nisa.memset(dst=mid_d.slice(dim=3, start=0, end=ho_lo, step=1), value=0.0)
                if ho_hi < H_out:
                    nisa.memset(dst=mid_d.slice(dim=3, start=ho_hi, end=H_out, step=1), value=0.0)
                mid_dh = mid_d.slice(dim=3, start=ho_lo, end=ho_hi, step=1)
                if wo_lo > 0:
                    nisa.memset(dst=mid_dh.slice(dim=4, start=0, end=wo_lo, step=1), value=0.0)
                if wo_hi < W_out:
                    nisa.memset(dst=mid_dh.slice(dim=4, start=wo_hi, end=W_out, step=1), value=0.0)
                in_d = do_lo * sd + kd - cfg.pad_d_left
                in_h = ho_lo * sh + kh - cfg.pad_h_top
                in_w = wo_lo * sw + kw - cfg.pad_w_left
                src = (
                    xplane5.slice(dim=1, start=0, end=nb, step=1)
                    .slice(dim=2, start=in_d, end=in_d + (do_hi - do_lo - 1) * sd + 1, step=sd)
                    .slice(dim=3, start=in_h, end=in_h + (ho_hi - ho_lo - 1) * sh + 1, step=sh)
                    .slice(dim=4, start=in_w, end=in_w + (wo_hi - wo_lo - 1) * sw + 1, step=sw)
                )
                dst = mid_dh.slice(dim=4, start=wo_lo, end=wo_hi, step=1)
                copy_d = do_hi - do_lo
                copy_h = ho_hi - ho_lo
                copy_w = wo_hi - wo_lo
                for copy_batch_idx in range(nb):
                    dst_batch = dst.slice(dim=1, start=copy_batch_idx, end=copy_batch_idx + 1, step=1)
                    src_batch = src.slice(dim=1, start=copy_batch_idx, end=copy_batch_idx + 1, step=1)
                    for copy_depth_idx in range(copy_d):
                        dst_plane = dst_batch.slice(dim=2, start=copy_depth_idx, end=copy_depth_idx + 1, step=1)
                        src_plane = src_batch.slice(dim=2, start=copy_depth_idx, end=copy_depth_idx + 1, step=1)
                        for copy_width_start in range(0, copy_w, VECTOR_COPY_F_MAX):
                            copy_width_len = min(VECTOR_COPY_F_MAX, copy_w - copy_width_start)
                            copy_height_step = max(1, VECTOR_COPY_F_MAX // copy_width_len)
                            for copy_height_start in range(0, copy_h, copy_height_step):
                                copy_height_len = min(copy_height_step, copy_h - copy_height_start)
                                dst_tile = dst_plane[
                                    :,
                                    :,
                                    :,
                                    copy_height_start : copy_height_start + copy_height_len,
                                    copy_width_start : copy_width_start + copy_width_len,
                                ]
                                src_tile = src_plane[
                                    :,
                                    :,
                                    :,
                                    copy_height_start : copy_height_start + copy_height_len,
                                    copy_width_start : copy_width_start + copy_width_len,
                                ]
                                nisa.tensor_copy(dst=dst_tile, src=src_tile, engine=nisa.vector_engine)

                xt3 = cur_xt[:, : n_pos_fast * ci_len].reshape((P_MAX, n_pos_fast, ci_len))
                t = 0

                while t < n_pt:
                    g = min(g_ci, n_pt - t)
                    sc = scratch[tp][:, : g * ci_len].reshape((P_MAX, g, ci_len))
                    drain_p = min(P_MAX, pack_pos - t * P_MAX)
                    for j in range(g):
                        p0 = (t + j) * P_MAX
                        ptile = min(P_MAX, pack_pos - p0)
                        nisa.nc_matmul(
                            dst=sc[:ptile, j, :ci_len],
                            stationary=cur_xs[:ci_len, p0 : p0 + ptile],
                            moving=identity[:ci_len, :ci_len],
                            accumulate=False,
                            is_transpose=True,
                        )
                    nisa.tensor_copy(
                        dst=xt3[:drain_p, t : t + g, :ci_len],
                        src=sc[:drain_p, :g, :ci_len],
                        engine=nisa.vector_engine
                        if copy_balance_count % VECTOR_COPY_INTERVAL == 0
                        else nisa.scalar_engine,
                    )
                    copy_balance_count += 1
                    tp = (tp + 1) % n_scr
                    t += g

                for chunk in _co_chunks(my_co, max(1, acc_width // co_tile)):
                    c0 = chunk[0]
                    c_last = chunk[-1]
                    col0 = c0 * co_tile
                    co_width = (c_last - c0) * co_tile + min(co_tile, C_out - c_last * co_tile)
                    cur_acc = acc_banks[acc_bank_idx]
                    for t in range(n_pt):
                        p0 = t * P_MAX
                        ptile = min(P_MAX, pack_pos - p0)
                        nisa.nc_matmul(
                            dst=cur_acc[:ci_len, :co_width],
                            stationary=xt3[:ptile, t, :ci_len],
                            moving=yt3[:ptile, t, col0 : col0 + co_width],
                            accumulate=(t > 0),
                        )
                    contiguous = c_last - c0 + 1 == len(chunk)
                    all_full = all(min(co_tile, C_out - c * co_tile) == co_tile for c in chunk)
                    _fold_eng = nisa.scalar_engine if tap_idx % 2 == 0 else nisa.vector_engine
                    if contiguous and all_full:
                        fold = dw_acc[k][:ci_len, tap_idx, c0 : c_last + 1, :].reshape((ci_len, co_width))
                        if b0 == b_lo:
                            nisa.tensor_copy(dst=fold, src=cur_acc[:ci_len, :co_width], engine=_fold_eng)
                        else:
                            nisa.tensor_tensor(dst=fold, data1=fold, data2=cur_acc[:ci_len, :co_width], op=nl.add)
                    else:
                        for c in chunk:
                            co_len = min(co_tile, C_out - c * co_tile)
                            off = (c - c0) * co_tile
                            fold = dw_acc[k][:ci_len, tap_idx, c, :co_len]
                            if b0 == b_lo:
                                nisa.tensor_copy(dst=fold, src=cur_acc[:ci_len, off : off + co_len], engine=_fold_eng)
                            else:
                                nisa.tensor_tensor(
                                    dst=fold, data1=fold, data2=cur_acc[:ci_len, off : off + co_len], op=nl.add
                                )
                    acc_bank_idx = (acc_bank_idx + 1) % len(acc_banks)

    if do_db:
        for c in all_co:
            if c not in db_co:
                continue
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            nisa.dma_copy(dst=db.slice(dim=0, start=co0, end=co0 + co_len, step=1), src=db_acc[:co_len, c])

    rbuf = 0
    for k in range(n_ci_local):
        ci_t = ci_ts[k]
        ci0 = ci_t * ci_tile
        ci_len = min(ci_tile, C_in - ci0)
        my_co = co_by_ci[ci_t]

        if batch_shard:
            other = 1 - cfg.prg_id
            nisa.sendrecv(
                src=dw_acc[k][:ci_len, :, :, :],
                dst=dw_recv[k][:ci_len, :, :, :],
                send_to_rank=other,
                recv_from_rank=other,
                pipe_id=0,
            )
            nisa.tensor_tensor(
                dst=dw_acc[k][:ci_len, :, :, :],
                data1=dw_acc[k][:ci_len, :, :, :],
                data2=dw_recv[k][:ci_len, :, :, :],
                op=nl.add,
            )
            if cfg.prg_id != 0:
                continue

        for tap_idx, (kd, kh, kw, _dl, _dh, _hl, _hh, _wl, _wh) in enumerate(taps):
            dw_tap = dw.select(dim=0, index=kd).select(dim=0, index=kh).select(dim=0, index=kw)
            # Collapse the per-C_out-tile write-out: dw_acc[k][:, tap, :, :] is contiguous.
            for chunk in _co_chunks(my_co, n_co):
                c0 = chunk[0]
                c_last = chunk[-1]
                all_full = all(min(co_tile, C_out - c * co_tile) == co_tile for c in chunk)
                res = result[rbuf]
                rbuf = (rbuf + 1) % 2
                if all_full:
                    col0 = c0 * co_tile
                    co_width = (c_last - c0 + 1) * co_tile
                    _store_via_result(
                        dw_tap.slice(dim=0, start=ci0, end=ci0 + ci_len, step=1).slice(
                            dim=1, start=col0, end=col0 + co_width, step=1
                        ),
                        dw_acc[k][:ci_len, tap_idx, c0 : c_last + 1, :].reshape((ci_len, co_width)),
                        res[:ci_len, :co_width],
                        engine=nisa.scalar_engine if rbuf % 2 == 0 else nisa.vector_engine,
                    )
                else:
                    for c in chunk:
                        co0 = c * co_tile
                        co_len = min(co_tile, C_out - co0)
                        res = result[rbuf]
                        rbuf = (rbuf + 1) % 2
                        _store_via_result(
                            dw_tap.slice(dim=0, start=ci0, end=ci0 + ci_len, step=1).slice(
                                dim=1, start=co0, end=co0 + co_len, step=1
                            ),
                            dw_acc[k][:ci_len, tap_idx, c, :co_len],
                            res[:ci_len, :co_len],
                            engine=nisa.scalar_engine if rbuf % 2 == 0 else nisa.vector_engine,
                        )

    if do_db:
        sbm.pop_heap()  # db_partial
        sbm.pop_heap()  # db_acc
    sbm.pop_heap()  # result[1]
    sbm.pop_heap()  # result[0]
    if dw_recv != None:
        for _ in dw_recv:
            sbm.pop_heap()  # dw_recv[k]
    for _ in dw_acc:
        sbm.pop_heap()  # dw_acc[k]
    for _ in xt:
        sbm.pop_heap()  # xt buffers
    for _ in xs:
        sbm.pop_heap()  # xs buffers
    sbm.pop_heap()  # yt
    for _ in y_plane:
        sbm.pop_heap()
    for _ in x_planes:
        sbm.pop_heap()
    sbm.pop_heap()  # identity
    return dw, db


def _dw_fast_prefetch(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager):
    """Allocate the FAST_PLANE dw identity + y_plane buffers and prefetch the first pack's dy planes into them, BEFORE the dx phase runs (and before dx's SBUF heap is freed)."""
    P_MAX = nl.tile_size.pmax
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D, H, W = cfg.D, cfg.H, cfg.W
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_in = D * H * W
    spatial_out = D_out * H_out * W_out

    ci_tile = min(P_MAX, C_in)
    co_tile = min(P_MAX, C_out)
    n_ci = div_ceil(C_in, ci_tile)
    n_co = div_ceil(C_out, co_tile)

    grid_can_shard = (n_ci * n_co) >= cfg.n_prgs and (n_ci * n_co) > 1
    batch_shard = cfg.is_sharded and not grid_can_shard
    if batch_shard:
        b_per = div_ceil(cfg.B, cfg.n_prgs)
        b_lo = min(b_per * cfg.prg_id, cfg.B)
        b_hi = min(b_lo + b_per, cfg.B)
    else:
        b_lo, b_hi = 0, cfg.B
    B_local = b_hi - b_lo

    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_co)]
    if grid_can_shard and not batch_shard and n_ci >= cfg.n_prgs:
        my_ci = list(range(n_ci))[cfg.prg_id :: cfg.n_prgs]
        my_blocks = [(i, c) for i in my_ci for c in range(n_co)]
    elif grid_can_shard:
        my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs]
    else:
        my_blocks = all_blocks
    all_co = sorted({c for (_i, c) in my_blocks})

    ds = sizeinbytes(dtype)
    per_batch_bytes = (spatial_in + n_co * spatial_out + n_co * co_tile + 2 * spatial_out + 2 * ci_tile) * ds
    # The identity + y_plane reserve sits ABOVE dx's SBUF region, so it permanently shrinks dx's budget.
    free_total = max(0, sbm.get_free_space() - _SBUF_HEADROOM)
    free_room = free_total // 2
    fast_pack = max(1, min(B_local, free_room // max(1, per_batch_bytes)))
    fast_L = fast_pack * spatial_out
    n_pos_fast = div_ceil(fast_L, P_MAX)
    # get_free_space() is per-partition (free-axis) bytes, so size the reserve the same way:
    reserve_bytes = (P_MAX + n_co * fast_L) * ds
    dx_min_bytes = cfg.n_taps * n_co * C_in * ds
    if reserve_bytes + dx_min_bytes > free_total:
        return None

    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwf_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)

    y_plane = [sbm.alloc_heap(shape=(co_tile, fast_L), dtype=dtype, name=f"dwf_yplane{c}") for c in range(n_co)]

    # Prefetch the first pack's dy planes (same DMA as the per-pack loop in _conv3d_dw_fast).
    y_flat = dy.flatten_dims(2, 4)
    b0 = b_lo
    nb = min(fast_pack, b_hi - b0)
    for c in all_co:
        if nb <= 0:
            break
        co0 = c * co_tile
        co_len = min(co_tile, C_out - co0)
        y_dst = y_plane[c][:co_len].reshape((co_len, fast_pack, spatial_out)).slice(dim=1, start=0, end=nb, step=1)
        nisa.dma_copy(
            dst=y_dst,
            src=y_flat.ap(
                pattern=[[spatial_out, co_len], [C_out * spatial_out, nb], [1, spatial_out]],
                offset=b0 * C_out * spatial_out + co0 * spatial_out,
            ),
            dge_mode=nisa.dge_mode.none,
        )
    return {
        "identity": identity,
        "y_plane": y_plane,
        "fast_pack": fast_pack,
        "fast_L": fast_L,
        "n_pos_fast": n_pos_fast,
    }


def _dw_fast_geom(cfg: "Conv3dBwdConfig", dtype, free_space: int, fast_pack_override=None) -> dict:
    """Deterministic dw-fast geometry (tiling, tap packing, batch shard, fast_pack)."""
    P_MAX = nl.tile_size.pmax
    C_in, C_out = cfg.C_in, cfg.C_out
    D, H, W = cfg.D, cfg.H, cfg.W
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_in = D * H * W
    spatial_out = D_out * H_out * W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w
    ci_tile = min(P_MAX, C_in)
    co_tile = min(P_MAX, C_out)
    n_ci = div_ceil(C_in, ci_tile)
    n_co = div_ceil(C_out, co_tile)
    taps = []
    for kd in range(cfg.K_d):
        do_lo, do_hi = _valid_output_range(D_out, D, kd, sd, 1, cfg.pad_d_left)
        for kh in range(cfg.K_h):
            ho_lo, ho_hi = _valid_output_range(H_out, H, kh, sh, 1, cfg.pad_h_top)
            for kw in range(cfg.K_w):
                wo_lo, wo_hi = _valid_output_range(W_out, W, kw, sw, 1, cfg.pad_w_left)
                taps.append((kd, kh, kw, do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi))
    n_taps = len(taps)
    pack_n = 2 if (ci_tile * 2 <= P_MAX and n_taps >= 2) else 1
    _active_idx = [
        ti for ti, (kd, kh, kw, dl, dh, hl, hh, wl, wh) in enumerate(taps) if dh > dl and hh > hl and wh > wl
    ]
    tap_groups = [_active_idx[i : i + pack_n] for i in range(0, len(_active_idx), pack_n)]
    tap_half = [0] * n_taps
    for _g in tap_groups:
        for _h, _ti in enumerate(_g):
            tap_half[_ti] = _h
    grid_can_shard = (n_ci * n_co) >= cfg.n_prgs and (n_ci * n_co) > 1
    batch_shard = cfg.is_sharded and not grid_can_shard
    if batch_shard:
        b_per = div_ceil(cfg.B, cfg.n_prgs)
        b_lo = min(b_per * cfg.prg_id, cfg.B)
        b_hi = min(b_lo + b_per, cfg.B)
    else:
        b_lo, b_hi = 0, cfg.B
    B_local = b_hi - b_lo
    if fast_pack_override != None:
        fast_pack = fast_pack_override
    else:
        ds = sizeinbytes(dtype)
        per_batch_bytes = (spatial_in + n_co * spatial_out + n_co * co_tile + 2 * spatial_out + ci_tile) * ds
        free_room = max(0, (free_space - _SBUF_HEADROOM) // 3)
        fast_pack = max(1, min(B_local, free_room // max(1, per_batch_bytes)))
        if B_local >= 2:
            fast_pack = min(fast_pack, div_ceil(B_local, 2))
    fast_L = fast_pack * spatial_out
    n_pos_fast = div_ceil(fast_L, P_MAX)
    return {
        "ci_tile": ci_tile,
        "co_tile": co_tile,
        "n_ci": n_ci,
        "n_co": n_co,
        "taps": taps,
        "n_taps": n_taps,
        "pack_n": pack_n,
        "tap_groups": tap_groups,
        "tap_half": tap_half,
        "grid_can_shard": grid_can_shard,
        "batch_shard": batch_shard,
        "b_lo": b_lo,
        "b_hi": b_hi,
        "B_local": B_local,
        "fast_pack": fast_pack,
        "fast_L": fast_L,
        "n_pos_fast": n_pos_fast,
        "spatial_in": spatial_in,
        "spatial_out": spatial_out,
    }


def _dw_fast_reserve(cfg: "Conv3dBwdConfig", dy, x_in, sbm: SbufManager) -> dict:
    """Reserve dw-fast's first-pack x_plane slot0 + yt buffers BEFORE dx runs."""
    P_MAX = nl.tile_size.pmax
    dtype = x_in.dtype
    geom = _dw_fast_geom(cfg, dtype, sbm.get_free_space())
    pack_n = geom["pack_n"]
    ci_tile = geom["ci_tile"]
    co_tile = geom["co_tile"]
    fast_pack = geom["fast_pack"]
    n_pos_fast = geom["n_pos_fast"]
    n_co = geom["n_co"]
    spatial_in = geom["spatial_in"]
    x_plane0 = sbm.alloc_heap(shape=(pack_n * ci_tile, fast_pack * spatial_in), dtype=dtype, name="dwf_xplane0")
    yt = [
        sbm.alloc_heap(shape=(P_MAX, n_pos_fast * co_tile), dtype=dtype, name=f"dwf_yt{c}", align=32)
        for c in range(n_co)
    ]
    return {"fast_pack": fast_pack, "x_plane0": x_plane0, "yt": yt, "geom": geom, "n_pop": n_co + 1}


def _dw_fast_prologue_load(cfg: "Conv3dBwdConfig", dy, x_in, prologue: dict) -> None:
    """Issue the hoisted first-pack x_plane load + yt dma_transpose into the pre-reserved (non-aliasing) buffers so the compiler can overlap them with the dx matmul tail."""
    P_MAX = nl.tile_size.pmax
    geom = prologue["geom"]
    ci_tile = geom["ci_tile"]
    co_tile = geom["co_tile"]
    n_co = geom["n_co"]
    n_ci = geom["n_ci"]
    fast_pack = geom["fast_pack"]
    n_pos_fast = geom["n_pos_fast"]
    pack_n = geom["pack_n"]
    spatial_in = geom["spatial_in"]
    spatial_out = geom["spatial_out"]
    b_lo = geom["b_lo"]
    b_hi = geom["b_hi"]
    grid_can_shard = geom["grid_can_shard"]
    C_in, C_out = cfg.C_in, cfg.C_out
    x_plane0 = prologue["x_plane0"]
    yt = prologue["yt"]
    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_co)]
    my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs] if grid_can_shard else all_blocks
    ci_ts = sorted({b[0] for b in my_blocks})
    packs = list(range(b_lo, b_hi, fast_pack))
    if not ci_ts or not packs:
        return
    ci_t = ci_ts[0]
    ci0 = ci_t * ci_tile
    ci_len = min(ci_tile, C_in - ci0)
    my_co = [c for (i, c) in my_blocks if i == ci_t]
    x_flat = x_in.flatten_dims(2, 4)
    y_flat = dy.flatten_dims(2, 4)
    b0 = packs[0]
    nb = min(fast_pack, b_hi - b0)
    for j in range(nb):
        nisa.dma_copy(
            dst=x_plane0[:ci_len, j * spatial_in : (j + 1) * spatial_in],
            src=x_flat.select(dim=0, index=b0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
        )
        if pack_n == 2:
            nisa.dma_copy(
                dst=x_plane0[ci_len : 2 * ci_len, j * spatial_in : (j + 1) * spatial_in],
                src=x_flat.select(dim=0, index=b0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
            )
    pack_pos = nb * spatial_out
    n_pt = div_ceil(pack_pos, P_MAX)
    for c in my_co:
        co0 = c * co_tile
        co_len = min(co_tile, C_out - co0)
        yt3 = yt[c][:, : n_pos_fast * co_len].reshape((P_MAX, n_pos_fast, co_len))
        for t in range(n_pt):
            p0 = t * P_MAX
            ptile = min(P_MAX, pack_pos - p0)
            off = 0
            while off < ptile:
                g_pos = p0 + off
                jb = g_pos // spatial_out
                sp_a = g_pos - jb * spatial_out
                seg = min(ptile - off, spatial_out - sp_a)
                nisa.dma_transpose(
                    dst=yt3[off : off + seg, t, :co_len],
                    src=y_flat.select(dim=0, index=b0 + jb)
                    .slice(dim=0, start=co0, end=co0 + co_len, step=1)
                    .slice(dim=1, start=sp_a, end=sp_a + seg, step=1),
                )
                off += seg


def _conv3d_dw_fast_packed(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, x_in: nl.NkiTensor, sbm: SbufManager, prologue=None
) -> nl.NkiTensor:
    """Packed-tap resident-plane dw for multi-tap / true-3D (FAST_PLANE, PACKED variant)."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    dtype = x_in.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D, H, W = cfg.D, cfg.H, cfg.W
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    spatial_in = D * H * W
    spatial_out = D_out * H_out * W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w

    geom = prologue["geom"] if prologue != None else _dw_fast_geom(cfg, dtype, sbm.get_free_space())
    ci_tile = geom["ci_tile"]
    co_tile = geom["co_tile"]
    n_ci = geom["n_ci"]
    n_co = geom["n_co"]
    taps = geom["taps"]
    n_taps = geom["n_taps"]
    pack_n = geom["pack_n"]
    tap_groups = geom["tap_groups"]
    tap_half = geom["tap_half"]
    grid_can_shard = geom["grid_can_shard"]
    batch_shard = geom["batch_shard"]
    b_lo = geom["b_lo"]
    b_hi = geom["b_hi"]
    B_local = geom["B_local"]
    fast_pack = geom["fast_pack"]
    fast_L = geom["fast_L"]
    n_pos_fast = geom["n_pos_fast"]

    dw = nl.ndarray(shape=(cfg.K_d, cfg.K_h, cfg.K_w, C_in, C_out), dtype=dtype, buffer=nl.shared_hbm)

    all_blocks = [(i, c) for i in range(n_ci) for c in range(n_co)]
    my_blocks = all_blocks[cfg.prg_id :: cfg.n_prgs] if grid_can_shard else all_blocks
    prologue_ci_t = sorted({b[0] for b in my_blocks})[0] if (prologue != None and my_blocks) else None

    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dwf_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)

    _NPB = 2  # x_plane / y_plane ping-pong depth
    # When two taps are packed (pack_n==2) we duplicate the ci channels onto the high partitions.
    if prologue != None:
        x_plane = [
            prologue["x_plane0"],
            sbm.alloc_heap(shape=(pack_n * ci_tile, fast_pack * spatial_in), dtype=dtype, name="dwf_xplane1"),
        ]
        yt = prologue["yt"]
    else:
        x_plane = [
            sbm.alloc_heap(shape=(pack_n * ci_tile, fast_pack * spatial_in), dtype=dtype, name=f"dwf_xplane{s}")
            for s in range(_NPB)
        ]
        yt = [
            sbm.alloc_heap(shape=(P_MAX, n_pos_fast * co_tile), dtype=dtype, name=f"dwf_yt{c}", align=32)
            for c in range(n_co)
        ]
    xs_bufs = [sbm.alloc_heap(shape=(pack_n * ci_tile, fast_L), dtype=dtype, name=f"dwf_xs{_k}") for _k in range(2)]
    xt = sbm.alloc_heap(shape=(P_MAX, n_pos_fast * pack_n * ci_tile), dtype=dtype, name="dwf_xt", align=32)
    dw_acc = sbm.alloc_heap(shape=(pack_n * ci_tile, n_taps, n_co, co_tile), dtype=_ACC_DTYPE, name="dwf_acc")
    dw_recv = (
        sbm.alloc_heap(shape=(pack_n * ci_tile, n_taps, n_co, co_tile), dtype=_ACC_DTYPE, name="dwf_recv")
        if batch_shard
        else None
    )
    result = sbm.alloc_heap(shape=(pack_n * ci_tile, co_tile), dtype=dtype, name="dwf_result")

    psum = _psum_offsets(cfg, "dw_fast")
    acc_off = psum["dw_fast_acc"][0]
    acc = nl.ndarray(shape=(pack_n * ci_tile, co_tile), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, acc_off))
    scratch = [
        nl.ndarray(shape=(P_MAX, F_MAX), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
        for off in psum["dw_fast_tp"]
    ]
    n_scr = len(scratch)

    x_flat = x_in.flatten_dims(2, 4)
    y_flat = dy.flatten_dims(2, 4)

    for ci_t in sorted({b[0] for b in my_blocks}):
        ci0 = ci_t * ci_tile
        ci_len = min(ci_tile, C_in - ci0)
        my_co = [c for (i, c) in my_blocks if i == ci_t]

        nisa.memset(dst=dw_acc[: pack_n * ci_len, :, :, :], value=0.0)
        g_ci = max(1, F_MAX // ci_len)

        tp = 0
        xs_par = 0
        packs = list(range(b_lo, b_hi, fast_pack))
        is_prologue_ci = prologue != None and ci_t == prologue_ci_t

        def _load_pack(k: int, slot: int) -> None:
            pb0 = packs[k]
            pnb = min(fast_pack, b_hi - pb0)
            for j in range(pnb):
                nisa.dma_copy(
                    dst=x_plane[slot][:ci_len, j * spatial_in : (j + 1) * spatial_in],
                    src=x_flat.select(dim=0, index=pb0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
                )
                if pack_n == 2:
                    nisa.dma_copy(
                        dst=x_plane[slot][ci_len : 2 * ci_len, j * spatial_in : (j + 1) * spatial_in],
                        src=x_flat.select(dim=0, index=pb0 + j).slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
                    )

        if packs and not is_prologue_ci:
            _load_pack(0, 0)
        for k, b0 in enumerate(packs):
            slot = k % _NPB
            if k + 1 < len(packs):
                _load_pack(k + 1, (k + 1) % _NPB)
            nb = min(fast_pack, b_hi - b0)
            pack_pos = nb * spatial_out
            n_pt = div_ceil(pack_pos, P_MAX)

            # Transpose dy straight from HBM into yt via the DMA engine (off the PE array).
            for c in my_co:
                co0 = c * co_tile
                co_len = min(co_tile, C_out - co0)
                yt3 = yt[c][:, : n_pos_fast * co_len].reshape((P_MAX, n_pos_fast, co_len))
                for t in range(n_pt):
                    p0 = t * P_MAX
                    ptile = min(P_MAX, pack_pos - p0)
                    off = 0
                    while off < ptile:
                        g_pos = p0 + off
                        jb = g_pos // spatial_out
                        sp_a = g_pos - jb * spatial_out
                        seg = min(ptile - off, spatial_out - sp_a)
                        if not (is_prologue_ci and k == 0):
                            nisa.dma_transpose(
                                dst=yt3[off : off + seg, t, :co_len],
                                src=y_flat.select(dim=0, index=b0 + jb)
                                .slice(dim=0, start=co0, end=co0 + co_len, step=1)
                                .slice(dim=1, start=sp_a, end=sp_a + seg, step=1),
                            )
                        off += seg

            for group in tap_groups:
                ng = len(group)
                pack_ci = ng * ci_len
                xs_cur = xs_bufs[xs_par]
                xs_par ^= 1
                for half, tap_idx in enumerate(group):
                    base_p = half * ci_len
                    xs5 = xs_cur[base_p : base_p + ci_len].reshape((ci_len, fast_pack, D_out, H_out, W_out))
                    (kd, kh, kw, do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi) = taps[tap_idx]
                    xsp = xs5.slice(dim=1, start=0, end=nb, step=1)
                    if do_lo > 0:
                        nisa.memset(dst=xsp.slice(dim=2, start=0, end=do_lo, step=1), value=0.0)
                    if do_hi < D_out:
                        nisa.memset(dst=xsp.slice(dim=2, start=do_hi, end=D_out, step=1), value=0.0)
                    mid_d = xsp.slice(dim=2, start=do_lo, end=do_hi, step=1)
                    if ho_lo > 0:
                        nisa.memset(dst=mid_d.slice(dim=3, start=0, end=ho_lo, step=1), value=0.0)
                    if ho_hi < H_out:
                        nisa.memset(dst=mid_d.slice(dim=3, start=ho_hi, end=H_out, step=1), value=0.0)
                    mid_dh = mid_d.slice(dim=3, start=ho_lo, end=ho_hi, step=1)
                    if wo_lo > 0:
                        nisa.memset(dst=mid_dh.slice(dim=4, start=0, end=wo_lo, step=1), value=0.0)
                    if wo_hi < W_out:
                        nisa.memset(dst=mid_dh.slice(dim=4, start=wo_hi, end=W_out, step=1), value=0.0)
                    in_d = do_lo * sd + kd - cfg.pad_d_left
                    in_h = ho_lo * sh + kh - cfg.pad_h_top
                    in_w = wo_lo * sw + kw - cfg.pad_w_left
                    xplane5h = x_plane[slot][base_p : base_p + ci_len].reshape((ci_len, fast_pack, D, H, W))
                    src = (
                        xplane5h.slice(dim=1, start=0, end=nb, step=1)
                        .slice(dim=2, start=in_d, end=in_d + (do_hi - do_lo - 1) * sd + 1, step=sd)
                        .slice(dim=3, start=in_h, end=in_h + (ho_hi - ho_lo - 1) * sh + 1, step=sh)
                        .slice(dim=4, start=in_w, end=in_w + (wo_hi - wo_lo - 1) * sw + 1, step=sw)
                    )
                    dst = mid_dh.slice(dim=4, start=wo_lo, end=wo_hi, step=1)
                    nisa.tensor_copy(dst=dst, src=src)

                # Single combined transpose of the packed (2*ci_len wide) shifted-x buffer.
                xt3 = xt[:, : n_pos_fast * pack_ci].reshape((P_MAX, n_pos_fast, pack_ci))
                g_pack = max(1, F_MAX // pack_ci)
                t = 0
                while t < n_pt:
                    g = min(g_pack, n_pt - t)
                    sc = scratch[tp][:, : g * pack_ci].reshape((P_MAX, g, pack_ci))
                    drain_p = min(P_MAX, pack_pos - t * P_MAX)
                    for j in range(g):
                        p0 = (t + j) * P_MAX
                        ptile = min(P_MAX, pack_pos - p0)
                        nisa.nc_matmul(
                            dst=sc[:ptile, j, :pack_ci],
                            stationary=xs_cur[:pack_ci, p0 : p0 + ptile],
                            moving=identity[:pack_ci, :pack_ci],
                            accumulate=False,
                        )
                    nisa.tensor_copy(
                        dst=xt3[:drain_p, t : t + g, :pack_ci],
                        src=sc[:drain_p, :g, :pack_ci],
                        engine=nisa.scalar_engine,
                    )
                    tp = (tp + 1) % n_scr
                    t += g

                xt3 = xt[:, : n_pos_fast * pack_ci].reshape((P_MAX, n_pos_fast, pack_ci))
                for c in my_co:
                    co_len = min(co_tile, C_out - c * co_tile)
                    yt3 = yt[c][:, : n_pos_fast * co_len].reshape((P_MAX, n_pos_fast, co_len))
                    for t in range(n_pt):
                        p0 = t * P_MAX
                        ptile = min(P_MAX, pack_pos - p0)
                        nisa.nc_matmul(
                            dst=acc[:pack_ci, :co_len],
                            stationary=xt3[:ptile, t, :pack_ci],
                            moving=yt3[:ptile, t, :co_len],
                            accumulate=(t > 0),
                        )
                    for half, tap_idx in enumerate(group):
                        hoff = half * ci_len
                        fold = dw_acc[hoff : hoff + ci_len, tap_idx, c, :co_len]
                        nisa.tensor_tensor(dst=fold, data1=fold, data2=acc[hoff : hoff + ci_len, :co_len], op=nl.add)

        if batch_shard:
            other = 1 - cfg.prg_id
            nisa.sendrecv(
                src=dw_acc[: pack_n * ci_len, :, :, :],
                dst=dw_recv[: pack_n * ci_len, :, :, :],
                send_to_rank=other,
                recv_from_rank=other,
                pipe_id=0,
            )
            nisa.tensor_tensor(
                dst=dw_acc[: pack_n * ci_len, :, :, :],
                data1=dw_acc[: pack_n * ci_len, :, :, :],
                data2=dw_recv[: pack_n * ci_len, :, :, :],
                op=nl.add,
            )
            if cfg.prg_id != 0:
                continue

        for tap_idx, (kd, kh, kw, _dl, _dh, _hl, _hh, _wl, _wh) in enumerate(taps):
            dw_tap = dw.select(dim=0, index=kd).select(dim=0, index=kh).select(dim=0, index=kw)
            hoff = tap_half[tap_idx] * ci_len
            for c in my_co:
                co0 = c * co_tile
                co_len = min(co_tile, C_out - co0)
                nisa.tensor_copy(
                    dst=result[hoff : hoff + ci_len, :co_len],
                    src=dw_acc[hoff : hoff + ci_len, tap_idx, c, :co_len],
                )
                nisa.dma_copy(
                    dst=dw_tap.slice(dim=0, start=ci0, end=ci0 + ci_len, step=1).slice(
                        dim=1, start=co0, end=co0 + co_len, step=1
                    ),
                    src=result[hoff : hoff + ci_len, :co_len],
                )

    sbm.pop_heap()  # result
    if dw_recv != None:
        sbm.pop_heap()  # dw_recv
    sbm.pop_heap()  # dw_acc
    sbm.pop_heap()  # xt
    for _ in xs_bufs:
        sbm.pop_heap()  # xs
    if prologue != None:
        # yt and x_plane[0] are prologue-owned (reserved before dx); pop only slot1 here.
        sbm.pop_heap()  # x_plane[1]
    else:
        for _ in yt:
            sbm.pop_heap()
        for _ in x_plane:
            sbm.pop_heap()
    sbm.pop_heap()  # identity
    return dw


def _conv3d_db(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, sbm: SbufManager) -> nl.NkiTensor:
    """Compute db[C_out] = sum over (batch, output positions) of dy."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    spatial = cfg.D_out * cfg.H_out * cfg.W_out
    c_tile = min(P_MAX, cfg.C_out)
    free_tile = min(spatial, F_MAX)

    db = nl.ndarray(shape=(cfg.C_out,), dtype=dy.dtype, buffer=nl.shared_hbm)
    input_buf = sbm.alloc_heap(shape=(c_tile, free_tile), dtype=dy.dtype, name="db_input")
    partial = sbm.alloc_heap(shape=(c_tile, 1), dtype=_ACC_DTYPE, name="db_partial")
    acc = sbm.alloc_heap(shape=(c_tile, 1), dtype=_ACC_DTYPE, name="db_acc")

    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, spatial]
    all_tiles = list(range(0, cfg.C_out, P_MAX))
    my_tiles = all_tiles[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_tiles

    for co0 in my_tiles:
        co_len = min(P_MAX, cfg.C_out - co0)
        nisa.memset(dst=acc[:co_len, :], value=0.0)
        for b in range(cfg.B):
            dyb = dy_flat.select(dim=0, index=b).slice(dim=0, start=co0, end=co0 + co_len, step=1)
            for sp0 in range(0, spatial, free_tile):
                sp_len = min(free_tile, spatial - sp0)
                nisa.dma_copy(
                    dst=input_buf[:co_len, :sp_len], src=dyb.slice(dim=1, start=sp0, end=sp0 + sp_len, step=1)
                )
                nisa.tensor_reduce(
                    dst=partial[:co_len, :], op=nl.add, data=input_buf[:co_len, :sp_len], axis=1, keepdims=True
                )
                nisa.tensor_tensor(dst=acc[:co_len, :], data1=acc[:co_len, :], data2=partial[:co_len, :], op=nl.add)
        nisa.dma_copy(dst=db.slice(dim=0, start=co0, end=co0 + co_len, step=1), src=acc[:co_len, 0])

    sbm.pop_heap()  # acc
    sbm.pop_heap()  # partial
    sbm.pop_heap()  # input_buf
    return db


def _dxg_alloc(cfg, dy, filters, sbm):
    """ALLOCATION: hoist identity, transposed Wpack[co,(kh,kw,ci)], the fixed width-scatter kernels, and dy/gt/dx buffers."""
    P_MAX, F_MAX = nl.tile_size.pmax, nl.tile_size.psum_fmax
    dtype = dy.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    H_out, W_out, W = cfg.H_out, cfg.W_out, cfg.W
    n_expand = cfg.K_h * cfg.K_w * C_in  # (kh,kw,ci) rows
    grp_max = min(P_MAX, n_expand)
    groups = []
    r = 0
    while r < n_expand:
        cnt = min((grp_max // C_in) * C_in, n_expand - r)
        groups.append((r, cnt))
        r += cnt

    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dxg_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)
    filt_flat = filters.select(dim=0, index=0).reshape((n_expand, C_out))  # K_d==1
    wsrc = sbm.alloc_heap(shape=(grp_max, C_out), dtype=dtype, name="dxg_wsrc")
    wpack = sbm.alloc_heap(shape=(C_out, n_expand), dtype=dtype, name="dxg_wpack")
    tp = nl.ndarray(
        shape=(P_MAX, P_MAX),
        dtype=_ACC_DTYPE,
        buffer=nl.psum,
        address=(0, (_NUM_PSUM_BANKS - 1) * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
    )
    for row0, cnt in groups:
        nisa.dma_copy(dst=wsrc[:cnt, :C_out], src=filt_flat.slice(dim=0, start=row0, end=row0 + cnt, step=1))
        _pe_transpose(wpack[:C_out, row0 : row0 + cnt], wsrc[:cnt, :C_out], tp, identity, cnt, C_out)

    skw = sbm.alloc_heap(shape=(W_out, cfg.K_w, W), dtype=dtype, name="dxg_skw")
    diff = sbm.alloc_heap(shape=(W_out, W), dtype=_ACC_DTYPE, name="dxg_diff")
    nisa.iota(dst=diff[:W_out, :W], pattern=[[1, W]], offset=0, channel_multiplier=-cfg.stride_w)
    for kw in range(cfg.K_w):
        nisa.tensor_scalar(
            dst=skw[:W_out, kw, :W], data=diff[:W_out, :W], op0=nl.equal, operand0=float(kw - cfg.pad_w_left)
        )

    iw_tile = min(P_MAX, W)
    my_b = list(range(cfg.prg_id, cfg.B, cfg.n_prgs)) if cfg.is_sharded else list(range(cfg.B))
    per_img = (W_out + H_out * n_expand + C_in) * sizeinbytes(dtype)
    b_blk = max(1, min(len(my_b), (sbm.get_free_space() - _SBUF_HEADROOM) // max(1, per_img)))

    # Batch G output rows (dx rows of a fixed ih%stride_h residue) onto the matmul moving free axis.
    G = max(1, F_MAX // (b_blk * C_in))
    # gt_all is padded in the ho dimension so out-of-range ho reads land on zeroed rows.
    PAD_LO = cfg.K_h
    PAD_HI = cfg.K_h
    H_pad = H_out + PAD_LO + PAD_HI
    chunk_p = (P_MAX // C_in) * C_in  # transpose store: cols processed per matmul (C_in-aligned)

    # Deepen the dy prefetch ring so several ho loads are in flight at once.
    DY_RING = 6
    dy_blk = [sbm.alloc_heap(shape=(C_out, b_blk * W_out), dtype=dtype, name=f"dxg_dyblk{d}") for d in range(DY_RING)]
    g_buf = sbm.alloc_heap(shape=(grp_max, W_out), dtype=dtype, name="dxg_g")
    gt_all = sbm.alloc_heap(shape=(W_out, H_pad, b_blk * n_expand), dtype=dtype, name="dxg_gt")
    # Per-kw accumulator: sum of the scatter moving operand over the valid kh taps.
    gt_sum = sbm.alloc_heap(shape=(W_out, cfg.K_w, G * b_blk * C_in), dtype=dtype, name="dxg_gtsum")
    dxrow = sbm.alloc_heap(shape=(iw_tile, G * b_blk * C_in), dtype=dtype, name="dxg_dxrow")

    psum = _psum_offsets(cfg, "dx_gemm")
    pp = len(psum["dx_g"])
    dxci = [sbm.alloc_heap(shape=(chunk_p, iw_tile), dtype=dtype, name=f"dxg_dxci{p}") for p in range(pp)]
    return {
        "groups": groups,
        "grp_max": grp_max,
        "n_expand": n_expand,
        "pp": pp,
        "b_blk": b_blk,
        "G": G,
        "PAD_LO": PAD_LO,
        "H_pad": H_pad,
        "chunk_p": chunk_p,
        "iw_tile": iw_tile,
        "my_b": my_b,
        "identity": identity,
        "wpack": wpack,
        "skw": skw,
        "dy_blk": dy_blk,
        "dy_ring": len(dy_blk),
        "g_buf": g_buf,
        "gt_all": gt_all,
        "gt_sum": gt_sum,
        "dxrow": dxrow,
        "dxci": dxci,
        "psum_g": [
            nl.ndarray(shape=(grp_max, W_out), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
            for off in psum["dx_g"]
        ],
        "tpg": [
            nl.ndarray(shape=(W_out, n_expand), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
            for off in psum["dx_tpg"]
        ],
        "psum_dx": nl.ndarray(
            shape=(iw_tile, G * b_blk * C_in), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, psum["dx_g"][0])
        ),
        "dxci_tp": [
            nl.ndarray(shape=(chunk_p, iw_tile), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
            for off in psum["dx_dxci_tp"]
        ],
    }


def _dxg_pass1(cfg, t, dy_flat, blk):
    """COMPUTE Pass 1: Gt[wo, (b, tap, ci)] for every output row = transpose of Wpack^T @ dy."""
    C_out, W_out, n_expand = cfg.C_out, cfg.W_out, t["n_expand"]
    PAD_LO, H_pad = t["PAD_LO"], t["H_pad"]
    # Zero the ho padding rows so out-of-range ho lookups in pass2 contribute nothing.
    for hp in range(PAD_LO):
        nisa.memset(dst=t["gt_all"][:W_out, hp, :], value=0.0)
    for hp in range(PAD_LO + cfg.H_out, H_pad):
        nisa.memset(dst=t["gt_all"][:W_out, hp, :], value=0.0)

    # Coalesce the per-batch dy loads for one ho into a single wide strided DMA.
    b_cnt = len(blk)
    n_pos = cfg.D_out * cfg.H_out * cfg.W_out
    bstep = (blk[1] - blk[0]) if b_cnt > 1 else 1
    batch_stride = bstep * C_out * n_pos

    def _load_dy(dst_buf, npos0):
        # dst columns are (j-major, wo-inner); match with a 3D source access pattern.
        src_view = dy_flat.ap(
            pattern=[[n_pos, C_out], [batch_stride, b_cnt], [1, W_out]],
            offset=blk[0] * C_out * n_pos + npos0,
        )
        nisa.dma_copy(dst=dst_buf[:C_out, : b_cnt * W_out], src=src_view)

    ring = t["dy_ring"]
    # Prologue: kick off the first (ring-1) ho loads so several are in flight before the main loop.
    for k in range(min(ring - 1, cfg.H_out)):
        _load_dy(t["dy_blk"][k % ring], k * W_out)
    for ho in range(cfg.H_out):
        cur = t["dy_blk"][ho % ring]
        nxt_ho = ho + ring - 1  # prefetch the ho that is (ring-1) ahead
        if nxt_ho < cfg.H_out:
            _load_dy(t["dy_blk"][nxt_ho % ring], nxt_ho * W_out)
        pp = 0
        # One matmul + one Vector drain over all n_expand columns per batch.
        for j in range(len(blk)):
            nisa.nc_matmul(
                dst=t["tpg"][pp][:W_out, :n_expand],
                stationary=cur[:C_out, j * W_out : j * W_out + W_out],
                moving=t["wpack"][:C_out, :n_expand],
                accumulate=False,
            )
            nisa.tensor_copy(
                dst=t["gt_all"][:W_out, ho + PAD_LO, j * n_expand : j * n_expand + n_expand],
                src=t["tpg"][pp][:W_out, :n_expand],
                engine=nisa.vector_engine,
            )
            pp = (pp + 1) % t["pp"]


def _dxg_pass2(cfg, t, dx_flat, blk):
    """COMPUTE+STORE Pass 2: batch G input rows of one ih%stride_h residue onto the moving free axis and scatter to dx."""
    C_in, W, W_out = cfg.C_in, cfg.W, cfg.W_out
    n_expand, b_blk, iw_tile = t["n_expand"], t["b_blk"], t["iw_tile"]
    G, PAD_LO, H_pad = t["G"], t["PAD_LO"], t["H_pad"]
    stride_h, pad = cfg.stride_h, cfg.pad_h_top
    b_cnt = len(blk)
    ho_stride = b_blk * n_expand  # gt_all free stride between consecutive ho
    part_stride = H_pad * ho_stride  # gt_all partition (wo) stride
    for r in range(stride_h):
        # kh taps that land on this ih residue -- all share the same one-hot skw[kw] stationary.
        khs = [kh for kh in range(cfg.K_h) if (r + pad - kh) % stride_h == 0]
        for ih0 in range(r, cfg.H, stride_h * G):
            g_cnt = min(G, (cfg.H - ih0 + stride_h - 1) // stride_h)
            n_col = g_cnt * b_cnt * C_in
            # Pre-sum the scatter moving operand over kh (vector engine), once per (r, ih0).
            for kw in range(cfg.K_w):
                for i, kh in enumerate(khs):
                    ho_start = (ih0 + pad - kh) // stride_h
                    tap = kh * cfg.K_w + kw
                    gt_view = t["gt_all"].ap(
                        pattern=[
                            [part_stride, W_out],
                            [n_expand, b_cnt],  # j (batch)
                            [1, C_in],  # ci (channel)
                            [ho_stride, g_cnt],  # g (innermost -> contiguous per (j,ci))
                        ],
                        offset=(ho_start + PAD_LO) * ho_stride + tap * C_in,
                    )
                    if i == 0:
                        nisa.tensor_copy(dst=t["gt_sum"][:W_out, kw, :n_col], src=gt_view)
                    else:
                        nisa.tensor_tensor(
                            dst=t["gt_sum"][:W_out, kw, :n_col],
                            data1=t["gt_sum"][:W_out, kw, :n_col],
                            data2=gt_view,
                            op=nl.add,
                        )
            for iw0 in range(0, W, iw_tile):
                iw_len = min(iw_tile, W - iw0)
                wrote = False
                if khs:
                    for kw in range(cfg.K_w):
                        nisa.nc_matmul(
                            dst=t["psum_dx"][:iw_len, :n_col],
                            stationary=t["skw"][:W_out, kw, iw0 : iw0 + iw_len],
                            moving=t["gt_sum"][:W_out, kw, :n_col],
                            accumulate=wrote,
                        )
                        wrote = True
                if wrote:
                    nisa.tensor_copy(dst=t["dxrow"][:iw_len, :n_col], src=t["psum_dx"][:iw_len, :n_col])
                else:
                    nisa.memset(dst=t["dxrow"][:iw_len, :n_col], value=0.0)
                spp = 0
                DHW = cfg.D * cfg.H * W
                chunk_p = t["chunk_p"]
                # dxrow columns are ordered (j, ci, g) with g innermost.
                n_unit = b_cnt * C_in
                if g_cnt <= chunk_p:
                    # Pack whole (j,ci) units into each transpose; still one dma per unit.
                    units_per_chunk = max(1, chunk_p // g_cnt)
                    for u0 in range(0, n_unit, units_per_chunk):
                        nu = min(units_per_chunk, n_unit - u0)
                        chunk = nu * g_cnt
                        col0 = u0 * g_cnt
                        nisa.nc_matmul(
                            dst=t["dxci_tp"][spp][:chunk, :iw_len],
                            stationary=t["dxrow"][:iw_len, col0 : col0 + chunk],
                            moving=t["identity"][:iw_len, :iw_len],
                            accumulate=False,
                        )
                        nisa.tensor_copy(dst=t["dxci"][spp][:chunk, :iw_len], src=t["dxci_tp"][spp][:chunk, :iw_len])
                        for uu in range(nu):
                            u = u0 + uu
                            j = u // C_in
                            ci = u % C_in
                            b = blk[j]
                            local = uu * g_cnt
                            dst_view = dx_flat.ap(
                                pattern=[[stride_h * W, g_cnt], [1, iw_len]],
                                offset=(b * C_in + ci) * DHW + ih0 * W + iw0,
                            )
                            nisa.dma_copy(dst=dst_view, src=t["dxci"][spp][local : local + g_cnt, :iw_len])
                        spp = (spp + 1) % t["pp"]
                else:
                    # g_cnt exceeds the transpose partition limit: split each unit's g into chunks.
                    for u in range(n_unit):
                        j = u // C_in
                        ci = u % C_in
                        b = blk[j]
                        for gs in range(0, g_cnt, chunk_p):
                            gl = min(chunk_p, g_cnt - gs)
                            col0 = u * g_cnt + gs
                            nisa.nc_matmul(
                                dst=t["dxci_tp"][spp][:gl, :iw_len],
                                stationary=t["dxrow"][:iw_len, col0 : col0 + gl],
                                moving=t["identity"][:iw_len, :iw_len],
                                accumulate=False,
                            )
                            nisa.tensor_copy(dst=t["dxci"][spp][:gl, :iw_len], src=t["dxci_tp"][spp][:gl, :iw_len])
                            dst_view = dx_flat.ap(
                                pattern=[[stride_h * W, gl], [1, iw_len]],
                                offset=(b * C_in + ci) * DHW + (ih0 + gs * stride_h) * W + iw0,
                            )
                            nisa.dma_copy(dst=dst_view, src=t["dxci"][spp][:gl, :iw_len])
                            spp = (spp + 1) % t["pp"]


def _dxg_free(sbm: SbufManager) -> None:
    # Keep this count synchronized with the SBUF allocations pushed by _dxg_alloc.
    heap_allocation_count = 13
    for _ in range(heap_allocation_count):
        sbm.pop_heap()


def _conv3d_dx_gemm(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager) -> nl.NkiTensor:
    """Tap-expand GEMM dx for the 2D stride-2 small-C_in stem (see _dx_gemm_supported)."""
    dx = nl.ndarray(shape=(cfg.B, cfg.C_in, cfg.D, cfg.H, cfg.W), dtype=dy.dtype, buffer=nl.shared_hbm)
    dx_flat = dx.reshape((cfg.B * cfg.C_in * cfg.D * cfg.H * cfg.W,))
    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, n_pos]
    t = _dxg_alloc(cfg, dy, filters, sbm)
    my_b, b_blk = t["my_b"], t["b_blk"]
    for blk0 in range(0, len(my_b), b_blk):
        blk = my_b[blk0 : blk0 + b_blk]
        _dxg_pass1(cfg, t, dy_flat, blk)
        _dxg_pass2(cfg, t, dx_flat, blk)
    _dxg_free(sbm)
    return dx


def _dxp_alloc(cfg, dy, filters, sbm):
    """ALLOCATION: transposed filters Wt[co,ci] (built once), dy pack buffer, dx drain."""
    P_MAX, F_MAX = nl.tile_size.pmax, nl.tile_size.psum_fmax
    dtype, C_in, C_out = filters.dtype, cfg.C_in, cfg.C_out
    ci_tile, co_tile, n_ci, n_co = cfg.ci_tile, cfg.co_tile, cfg.n_ci, cfg.n_co
    sp_tile = min(cfg.spatial_in, F_MAX)
    imgs_per_pack = max(1, F_MAX // sp_tile)
    pack_w = imgs_per_pack * sp_tile
    filt2d = filters.select(dim=0, index=0).select(dim=0, index=0).select(dim=0, index=0)
    wt = [
        [sbm.alloc_heap(shape=(co_tile, ci_tile), dtype=dtype, name=f"dxp_wt{c}_{i}", align=32) for i in range(n_ci)]
        for c in range(n_co)
    ]
    for c in range(n_co):
        co0 = c * co_tile
        co_len = min(co_tile, C_out - co0)
        for i in range(n_ci):
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            nisa.dma_transpose(
                dst=wt[c][i][:co_len, :ci_len],
                src=filt2d.slice(dim=0, start=ci0, end=ci0 + ci_len, step=1).slice(
                    dim=1, start=co0, end=co0 + co_len, step=1
                ),
            )
    dy_buf = [sbm.alloc_heap(shape=(co_tile, pack_w), dtype=dtype, name=f"dxp_dy{c}") for c in range(n_co)]
    out_buf = sbm.alloc_heap(shape=(ci_tile, pack_w), dtype=dtype, name="dxp_out")
    mm_banks = _psum_offsets(cfg, "dx_pointwise")["dxp_mm"]
    return {
        "wt": wt,
        "dy_buf": dy_buf,
        "out_buf": out_buf,
        "sp_tile": sp_tile,
        "imgs_per_pack": imgs_per_pack,
        "mm_banks": mm_banks,
    }


def _dxp_load_dy(cfg, t, dy_flat, batches, sp0, sp):
    """LOAD: pack this group's batches' dy [co, (b, sp)] side by side on the moving axis."""
    for c in range(cfg.n_co):
        co0 = c * cfg.co_tile
        co_len = min(cfg.co_tile, cfg.C_out - co0)
        for j, b in enumerate(batches):
            nisa.dma_copy(
                dst=t["dy_buf"][c][:co_len, j * sp : j * sp + sp],
                src=dy_flat.select(dim=0, index=b)
                .slice(dim=0, start=co0, end=co0 + co_len, step=1)
                .slice(dim=1, start=sp0, end=sp0 + sp, step=1),
            )


def _dxp_compute_store(cfg, t, dx_flat, batches, sp0, sp):
    """COMPUTE dx=sum_co Wt^T@dy (per ci tile) + STORE each image's slice to HBM."""
    w = len(batches) * sp
    n_mm = len(t["mm_banks"])
    for i in range(cfg.n_ci):
        ci0 = i * cfg.ci_tile
        ci_len = min(cfg.ci_tile, cfg.C_in - ci0)
        dx_psum = nl.ndarray(shape=(ci_len, w), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, t["mm_banks"][i % n_mm]))
        for c in range(cfg.n_co):
            co_len = min(cfg.co_tile, cfg.C_out - c * cfg.co_tile)
            nisa.nc_matmul(
                dst=dx_psum,
                stationary=t["wt"][c][i][:co_len, :ci_len],
                moving=t["dy_buf"][c][:co_len, :w],
                accumulate=(c > 0),
            )
        nisa.tensor_copy(
            dst=t["out_buf"][:ci_len, :w], src=dx_psum, engine=nisa.scalar_engine if i % 2 == 0 else nisa.vector_engine
        )
        for j, b in enumerate(batches):
            nisa.dma_copy(
                dst=dx_flat.select(dim=0, index=b)
                .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
                .slice(dim=1, start=sp0, end=sp0 + sp, step=1),
                src=t["out_buf"][:ci_len, j * sp : j * sp + sp],
            )


def _dxp_free(sbm, t):
    sbm.pop_heap()  # out_buf
    for _ in t["dy_buf"]:
        sbm.pop_heap()
    for row in t["wt"]:
        for _ in row:
            sbm.pop_heap()  # wt


def _conv3d_dx_pointwise(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager
) -> nl.NkiTensor:
    """Direct 1x1x1 (stride 1) dx: dx[b,ci,p] = sum_co W[ci,co]*dy[b,co,p] (plain C_out contraction)."""
    dx = nl.ndarray(shape=(cfg.B, cfg.C_in, cfg.D, cfg.H, cfg.W), dtype=filters.dtype, buffer=nl.shared_hbm)
    dx_flat = dx.flatten_dims(2, 4)
    dy_flat = dy.flatten_dims(2, 4)
    t = _dxp_alloc(cfg, dy, filters, sbm)
    my_b = list(range(cfg.prg_id, cfg.B, cfg.n_prgs)) if cfg.is_sharded else list(range(cfg.B))
    ipp = t["imgs_per_pack"]
    for gi in range(0, len(my_b), ipp):
        batches = my_b[gi : gi + ipp]
        for sp0 in range(0, cfg.spatial_in, t["sp_tile"]):
            sp = min(t["sp_tile"], cfg.spatial_in - sp0)
            _dxp_load_dy(cfg, t, dy_flat, batches, sp0, sp)
            _dxp_compute_store(cfg, t, dx_flat, batches, sp0, sp)
    _dxp_free(sbm, t)
    return dx


def _dxg2_fill_dy(
    dy_flat,
    b,
    dyp_buf,
    dyc_buf,
    n_co,
    co_tile,
    C_out,
    H_pad,
    W_pad,
    r_dy0,
    c_dy0,
    H_out,
    W_out,
    pack_ok,
    dlw,
    db,
    db_acc,
    db_tmp,
):
    """Fill one (double-buffered) dy_pad/dy_ctg set for batch ``b``."""
    dyb = dy_flat.select(dim=0, index=b)  # [C_out, H_out*W_out]
    for c in range(n_co):
        co0 = c * co_tile
        co_len = min(co_tile, C_out - co0)
        nisa.dma_copy(
            dst=dyc_buf[c][:co_len, :],
            src=dyb.slice(dim=0, start=co0, end=co0 + co_len, step=1),
        )
        if db != None:
            nisa.tensor_reduce(
                dst=db_tmp[:co_len, c : c + 1],
                data=dyc_buf[c][:co_len, :],
                op=nl.add,
                axis=1,
                keepdims=True,
            )
            nisa.tensor_tensor(
                dst=db_acc[:co_len, c : c + 1],
                data1=db_acc[:co_len, c : c + 1],
                data2=db_tmp[:co_len, c : c + 1],
                op=nl.add,
            )
        dyp3 = dyp_buf[c][:co_len, :].reshape((co_len, H_pad, W_pad))
        dst_in = dyp3.slice(dim=1, start=r_dy0, end=r_dy0 + H_out, step=1).slice(
            dim=2, start=c_dy0, end=c_dy0 + W_out, step=1
        )
        nisa.tensor_copy(
            dst=dst_in,
            src=dyc_buf[c][:co_len, :].reshape((co_len, H_out, W_out)),
            engine=nisa.vector_engine,
        )
        if pack_ok:
            nisa.dma_copy(
                dst=dyc_buf[c][co_tile : co_tile + co_len, :],
                src=dyb.slice(dim=0, start=co0, end=co0 + co_len, step=1),
            )
            dyp3t = dyp_buf[c][co_tile : co_tile + co_len, :].reshape((co_len, H_pad, W_pad))
            dst_in_t = dyp3t.slice(dim=1, start=r_dy0, end=r_dy0 + H_out, step=1).slice(
                dim=2, start=c_dy0 - dlw, end=c_dy0 - dlw + W_out, step=1
            )
            nisa.tensor_copy(
                dst=dst_in_t,
                src=dyc_buf[c][co_tile : co_tile + co_len, :].reshape((co_len, H_out, W_out)),
                engine=nisa.vector_engine,
            )


def _conv3d_dx_gemm2d(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager, db: nl.NkiTensor = None
) -> nl.NkiTensor:
    """Stride-1 2D dx as an im2col-over-dy GEMM."""
    F_MAX = nl.tile_size.psum_fmax
    dtype, C_in, C_out = filters.dtype, cfg.C_in, cfg.C_out
    H, W, H_out, W_out = cfg.H, cfg.W, cfg.H_out, cfg.W_out
    dlh, dlw = cfg.dilation_h, cfg.dilation_w
    K_h, K_w = cfg.K_h, cfg.K_w
    co_tile, ci_tile, n_co, n_ci = cfg.co_tile, cfg.ci_tile, cfg.n_co, cfg.n_ci
    dx = nl.ndarray(shape=(cfg.B, cfg.C_in, cfg.D, cfg.H, cfg.W), dtype=dtype, buffer=nl.shared_hbm)
    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, H_out*W_out]

    # Transposed filters Wt[tap][c] = [co, ci].
    filt_flat = filters.reshape((cfg.K_d * cfg.K_h * cfg.K_w, C_in, C_out))
    wt = _build_wt_per_tap(sbm, filt_flat, cfg.n_taps, C_in, C_out, co_tile, n_co, dtype, "dxg2_wt")

    # --- Tap-pair packing: stack two C_out=64 tap contractions into one 128-row matmul ---
    _c_dy0 = (K_w - 1) * dlw - cfg.pad_w_left
    pack_ok = (n_co == 1) and (co_tile <= 64) and (K_w >= 2) and (_c_dy0 - dlw >= 0)
    pairs = []
    if pack_ok:
        for kh in range(K_h):
            kw = K_w - 1
            while kw >= 0:
                if kw - 1 >= 0:
                    pairs.append(((kh, kw), (kh, kw - 1)))
                    kw -= 2
                else:
                    pairs.append(((kh, kw), None))
                    kw -= 1
    co_len0 = min(co_tile, C_out)
    wt_pk = []
    for pi, (A, B) in enumerate(pairs):
        if B == None:
            wt_pk.append(None)
            continue
        w = sbm.alloc_heap(shape=(128, C_in), dtype=dtype, name=f"dxg2_wtpk{pi}", align=32)
        for half, (kh, kw) in enumerate((A, B)):
            t = kh * K_w + kw
            ft = filt_flat.select(dim=0, index=t)
            p0 = half * co_tile
            nisa.dma_transpose(dst=w[p0 : p0 + co_len0, :C_in], src=ft.slice(dim=1, start=0, end=co_len0, step=1))
        wt_pk.append(w)
    n_wt_pk = sum(1 for _, B in pairs if B != None)
    pad_p = 128 if pack_ok else co_tile

    # Zero-padded dy buffer per c: [co, H_pad, W_pad]. Valid dy (ho,wo) lands at (r_dy0+ho, c_dy0+wo).
    H_pad = H + (K_h - 1) * dlh
    W_pad = W + (K_w - 1) * dlw
    r_dy0 = (K_h - 1) * dlh - cfg.pad_h_top
    c_dy0 = (K_w - 1) * dlw - cfg.pad_w_left
    my_b = list(range(cfg.prg_id, cfg.B, cfg.n_prgs)) if cfg.is_sharded else list(range(cfg.B))
    # Double-buffer dy_pad/dy_ctg so batch b+1's fill overlaps batch b's matmuls.
    NBUF = 2 if len(my_b) > 1 else 1
    dy_pad = [
        [sbm.alloc_heap(shape=(pad_p, H_pad * W_pad), dtype=dtype, name=f"dxg2_dy{buf}_{c}") for c in range(n_co)]
        for buf in range(NBUF)
    ]
    dy_ctg = [
        [sbm.alloc_heap(shape=(pad_p, H_out * W_out), dtype=dtype, name=f"dxg2_dyc{buf}_{c}") for c in range(n_co)]
        for buf in range(NBUF)
    ]
    band = max(1, F_MAX // W)
    out_sbuf = sbm.alloc_heap(shape=(ci_tile, band * W), dtype=dtype, name="dxg2_out")
    g_off = _psum_offsets(cfg, "dx_col2im")["dxc_g"][0]  # gemm2d uses one accumulator bank

    # Fused bias gradient: partial db[co] accumulated on the Vector engine from the dy copy.
    db_acc = db_tmp = None
    if db != None:
        db_acc = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dxg2_db_acc")
        db_tmp = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dxg2_db_tmp")
        nisa.memset(dst=db_acc[:, :], value=0.0)

    taps = [(kh, kw) for kh in range(K_h) for kw in range(K_w)]
    # Borders stay zero for the whole run; interiors are fully overwritten each batch.
    for buf in range(NBUF):
        for c in range(n_co):
            nisa.memset(dst=dy_pad[buf][c][:pad_p, :], value=0.0)
    fill_args = (
        n_co,
        co_tile,
        C_out,
        H_pad,
        W_pad,
        r_dy0,
        c_dy0,
        H_out,
        W_out,
        pack_ok,
        dlw,
        db,
        db_acc,
        db_tmp,
    )
    if my_b:
        _dxg2_fill_dy(dy_flat, my_b[0], dy_pad[0], dy_ctg[0], *fill_args)
    for idx, b in enumerate(my_b):
        cur = idx % NBUF
        # Prefill the next batch into the other buffer set so its fill overlaps.
        if idx + 1 < len(my_b):
            nxt = (idx + 1) % NBUF
            _dxg2_fill_dy(dy_flat, my_b[idx + 1], dy_pad[nxt], dy_ctg[nxt], *fill_args)
        for io0 in range(0, H, band):
            io1 = min(H, io0 + band)
            bh = io1 - io0
            free_n = bh * W
            for i in range(n_ci):
                ci0 = i * ci_tile
                ci_len = min(ci_tile, C_in - ci0)
                dx_psum = nl.ndarray(shape=(ci_len, free_n), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, g_off))
                first = True
                if pack_ok:
                    for pi, (A, B) in enumerate(pairs):
                        khA, kwA = A
                        r = io0 + (K_h - 1 - khA) * dlh
                        cwA = (K_w - 1 - kwA) * dlw
                        if B == None:
                            mov = (
                                dy_pad[cur][0][:co_len0, :]
                                .reshape((co_len0, H_pad, W_pad))
                                .slice(dim=1, start=r, end=r + bh, step=1)
                                .slice(dim=2, start=cwA, end=cwA + W, step=1)
                            )
                            tA = khA * K_w + kwA
                            nisa.nc_matmul(
                                dst=dx_psum,
                                stationary=wt[tA][0][:co_len0, ci0 : ci0 + ci_len],
                                moving=mov,
                                accumulate=(not first),
                            )
                        else:
                            mov = (
                                dy_pad[cur][0][: 2 * co_len0, :]
                                .reshape((2 * co_len0, H_pad, W_pad))
                                .slice(dim=1, start=r, end=r + bh, step=1)
                                .slice(dim=2, start=cwA, end=cwA + W, step=1)
                            )
                            nisa.nc_matmul(
                                dst=dx_psum,
                                stationary=wt_pk[pi][:, ci0 : ci0 + ci_len],
                                moving=mov,
                                accumulate=(not first),
                            )
                        first = False
                else:
                    for t_idx, (kh, kw) in enumerate(taps):
                        r = io0 + (K_h - 1 - kh) * dlh
                        cw = (K_w - 1 - kw) * dlw
                        for c in range(n_co):
                            co0 = c * co_tile
                            co_len = min(co_tile, C_out - co0)
                            mov = (
                                dy_pad[cur][c][:co_len, :]
                                .reshape((co_len, H_pad, W_pad))
                                .slice(dim=1, start=r, end=r + bh, step=1)
                                .slice(dim=2, start=cw, end=cw + W, step=1)
                            )
                            nisa.nc_matmul(
                                dst=dx_psum,
                                stationary=wt[t_idx][c][:co_len, ci0 : ci0 + ci_len],
                                moving=mov,
                                accumulate=(not first),
                            )
                            first = False
                nisa.tensor_copy(dst=out_sbuf[:ci_len, :free_n], src=dx_psum, engine=nisa.scalar_engine)
                dst_hbm = (
                    dx.select(dim=0, index=b)
                    .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
                    .select(dim=1, index=0)
                    .slice(dim=1, start=io0, end=io1, step=1)
                    .flatten_dims(1, 2)
                )
                nisa.dma_copy(dst=dst_hbm, src=out_sbuf[:ci_len, :free_n])
    if db != None:
        # Combine per-core partial db (each core summed only its own batches) then store.
        if cfg.is_sharded:
            other = 1 - cfg.prg_id
            db_recv = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dxg2_db_recv")
            nisa.sendrecv(src=db_acc[:, :], dst=db_recv[:, :], send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_tensor(dst=db_acc[:, :], data1=db_acc[:, :], data2=db_recv[:, :], op=nl.add)
        db_out = sbm.alloc_heap(shape=(co_tile, n_co), dtype=dy.dtype, name="dxg2_db_out")
        for c in range(n_co):
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            nisa.tensor_copy(dst=db_out[:co_len, c : c + 1], src=db_acc[:co_len, c : c + 1])
            nisa.dma_copy(dst=db.slice(dim=0, start=co0, end=co0 + co_len, step=1), src=db_out[:co_len, c])
        sbm.pop_heap()  # db_out
        if cfg.is_sharded:
            sbm.pop_heap()  # db_recv
        sbm.pop_heap()  # db_tmp
        sbm.pop_heap()  # db_acc
    sbm.pop_heap()  # out_sbuf
    for _ in range(n_co * NBUF):
        sbm.pop_heap()  # dy_ctg
    for _ in range(n_co * NBUF):
        sbm.pop_heap()  # dy_pad
    for _ in range(n_wt_pk):
        sbm.pop_heap()  # wt_pk
    for _ in range(cfg.n_taps):
        for _ in range(n_co):
            sbm.pop_heap()  # wt
    return dx


def _conv3d_dx_pointwise_strided(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager, fold_db: bool = False
):
    """1x1x1 stride>1 dx: dxc = sum_co W[ci,co]*dy over C_out, batch-packed GEMM + strided DMA-store."""
    P_MAX, F_MAX = nl.tile_size.pmax, nl.tile_size.psum_fmax
    dtype = filters.dtype
    C_in, C_out = cfg.C_in, cfg.C_out
    D, H, W = cfg.D, cfg.H, cfg.W
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w
    ci_tile, co_tile, n_ci, n_co = cfg.ci_tile, cfg.co_tile, cfg.n_ci, cfg.n_co
    spatial_out = D_out * H_out * W_out
    spatial_in = D * H * W

    dx = nl.ndarray(shape=(cfg.B, C_in, D, H, W), dtype=dtype, buffer=nl.shared_hbm)
    dx_flat = dx.flatten_dims(2, 4)  # [B, C_in, spatial_in]
    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, spatial_out]

    db = None
    db_part = db_tmp = None
    if fold_db:
        db = nl.ndarray(shape=(C_out,), dtype=dy.dtype, buffer=nl.shared_hbm)
        db_part = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dxps_db_part")
        db_tmp = sbm.alloc_heap(shape=(co_tile, 1), dtype=_ACC_DTYPE, name="dxps_db_tmp")
        nisa.memset(dst=db_part[:, :], value=0.0)

    imgs_per_pack = max(1, F_MAX // spatial_out)
    pack_w = imgs_per_pack * spatial_out
    db_scr = None
    if fold_db:
        db_scr = sbm.alloc_heap(shape=(co_tile, pack_w), dtype=_ACC_DTYPE, name="dxps_db_scr")

    # Transposed filters Wt[co, ci] (built once).
    filt2d = filters.select(dim=0, index=0).select(dim=0, index=0).select(dim=0, index=0)  # [C_in, C_out]
    wt = [
        [sbm.alloc_heap(shape=(co_tile, ci_tile), dtype=dtype, name=f"dxps_wt{c}_{i}", align=32) for i in range(n_ci)]
        for c in range(n_co)
    ]
    # Build the transposed filters on the (idle-at-startup) PE via nc_matmul with an identity.
    mm_banks_tp = _psum_offsets(cfg, "dx_pointwise")["dxp_mm"]
    n_mm_tp = len(mm_banks_tp)
    identity = sbm.alloc_heap(shape=(P_MAX, P_MAX), dtype=dtype, name="dxps_id")
    nl.shared_identity_matrix(P_MAX, dtype=dtype, dst=identity)
    f_stage = [sbm.alloc_heap(shape=(ci_tile, C_out), dtype=dtype, name=f"dxps_fstage{i}") for i in range(n_ci)]
    for i in range(n_ci):
        ci0 = i * ci_tile
        ci_len = min(ci_tile, C_in - ci0)
        nisa.dma_copy(
            dst=f_stage[i][:ci_len, :],
            src=filt2d.slice(dim=0, start=ci0, end=ci0 + ci_len, step=1),
        )
    for c in range(n_co):
        co0 = c * co_tile
        co_len = min(co_tile, C_out - co0)
        for i in range(n_ci):
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            tp_psum = nl.ndarray(
                shape=(co_len, ci_len),
                dtype=_ACC_DTYPE,
                buffer=nl.psum,
                address=(0, mm_banks_tp[(c * n_ci + i) % n_mm_tp]),
            )
            _pe_transpose(
                wt[c][i][:co_len, :ci_len], f_stage[i][:ci_len, co0 : co0 + co_len], tp_psum, identity, ci_len, co_len
            )
    dy_buf = [sbm.alloc_heap(shape=(co_tile, pack_w), dtype=dtype, name=f"dxps_dy{c}") for c in range(n_co)]
    # Reusable, pre-zeroed output planes; gap positions are memset once and never touched again.
    pack_in = imgs_per_pack * spatial_in
    N_PLANE = 8  # Rotate eight output planes to overlap pointwise dx compute and stores.
    planes = [sbm.alloc_heap(shape=(ci_tile, pack_in), dtype=dtype, name=f"dxps_plane{p}") for p in range(N_PLANE)]
    for p in range(N_PLANE):
        nisa.memset(dst=planes[p][:, :], value=0.0)
    mm_banks = _psum_offsets(cfg, "dx_pointwise")["dxp_mm"]
    n_mm = len(mm_banks)

    my_b = list(range(cfg.prg_id, cfg.B, cfg.n_prgs)) if cfg.is_sharded else list(range(cfg.B))
    batch_step = cfg.n_prgs if cfg.is_sharded else 1

    plane_sel = 0
    for gi in range(0, len(my_b), imgs_per_pack):
        batches = my_b[gi : gi + imgs_per_pack]
        nb = len(batches)
        w = nb * spatial_out
        base_b = batches[0]
        # Load this group's batches' dy [co, (b, spatial_out)] packed side by side in one DMA.
        for c in range(n_co):
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            nisa.dma_copy(
                dst=dy_buf[c][:co_len, :w].reshape((co_len, nb, spatial_out)),
                src=dy_flat.ap(
                    pattern=[[spatial_out, co_len], [batch_step * C_out * spatial_out, nb], [1, spatial_out]],
                    offset=base_b * C_out * spatial_out + co0 * spatial_out,
                ),
            )
        if fold_db:
            for c in range(n_co):
                co0 = c * co_tile
                co_len = min(co_tile, C_out - co0)
                # Reduce over the packed free dim on the Scalar (Activation) engine.
                nisa.activation(
                    dst=db_scr[:co_len, :w],
                    op=nl.copy,
                    data=dy_buf[c][:co_len, :w],
                    reduce_op=nl.add,
                    reduce_res=db_tmp[:co_len, :],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                nisa.tensor_tensor(
                    dst=db_part[:co_len, c : c + 1],
                    data1=db_part[:co_len, c : c + 1],
                    data2=db_tmp[:co_len, :],
                    op=nl.add,
                )
        for i in range(n_ci):
            ci0 = i * ci_tile
            ci_len = min(ci_tile, C_in - ci0)
            dx_psum = nl.ndarray(shape=(ci_len, w), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, mm_banks[i % n_mm]))
            for c in range(n_co):
                co_len = min(co_tile, C_out - c * co_tile)
                nisa.nc_matmul(
                    dst=dx_psum,
                    stationary=wt[c][i][:co_len, :ci_len],
                    moving=dy_buf[c][:co_len, :w],
                    accumulate=(c > 0),
                )
            plane = planes[plane_sel]
            plane_sel = (plane_sel + 1) % N_PLANE
            for j, b in enumerate(batches):
                # Scatter this batch's dxc straight from PSUM onto its slot's strided value grid.
                dst_grid = (
                    plane[:ci_len, j * spatial_in : j * spatial_in + spatial_in]
                    .reshape((ci_len, D, H, W))
                    .slice(dim=1, start=0, end=D_out * sd, step=sd)
                    .slice(dim=2, start=0, end=H_out * sh, step=sh)
                    .slice(dim=3, start=0, end=W_out * sw, step=sw)
                )
                nisa.tensor_copy(
                    dst=dst_grid,
                    src=dx_psum[:ci_len, j * spatial_out : j * spatial_out + spatial_out].reshape(
                        (ci_len, D_out, H_out, W_out)
                    ),
                    engine=nisa.scalar_engine if (i + j) % 2 == 0 else nisa.vector_engine,
                )
            # One coalesced strided DMA flushes this ci-tile for ALL nb batches at once.
            nisa.dma_copy(
                dst=dx_flat.ap(
                    pattern=[[spatial_in, ci_len], [batch_step * C_in * spatial_in, nb], [1, spatial_in]],
                    offset=base_b * C_in * spatial_in + ci0 * spatial_in,
                ),
                src=plane[:ci_len, : nb * spatial_in].reshape((ci_len, nb, spatial_in)),
            )

    for _ in range(N_PLANE):
        sbm.pop_heap()  # planes
    for _ in range(n_co):
        sbm.pop_heap()  # dy_buf
    for row in wt:
        for _ in row:
            sbm.pop_heap()  # wt

    if not fold_db:
        return dx

    # Defer the cross-core db reduction + write to kernel end (after the dw pass).
    sbm.pop_heap()  # db_scr
    sbm.pop_heap()  # db_tmp
    return dx, db, db_part


def _issue_db_sendrecv(cfg, db_part, sbm):
    """Kick off the cross-core exchange of the deferred per-shard db partial BEFORE the dw pass so the sendrecv rendezvous overlaps dw compute instead of stalling after it."""
    if not cfg.is_sharded:
        return None
    co_tile, n_co = cfg.co_tile, cfg.n_co
    other = 1 - cfg.prg_id
    db_recv = sbm.alloc_heap(shape=(co_tile, n_co), dtype=_ACC_DTYPE, name="dxps_db_recv")
    nisa.sendrecv(src=db_part, dst=db_recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
    return db_recv


def _finalize_db_pointwise_strided(cfg, dy, db, db_part, db_recv, sbm):
    """Complete the deferred db: consume the already-issued cross-core exchange (db_recv) and store."""
    co_tile, n_co, C_out = cfg.co_tile, cfg.n_co, cfg.C_out
    # Each core summed dy only over its batch shard; combine across the two cores.
    if cfg.is_sharded:
        nisa.tensor_tensor(dst=db_part[:, :], data1=db_part[:, :], data2=db_recv[:, :], op=nl.add)
        sbm.pop_heap()  # db_recv
    if not cfg.is_sharded or cfg.prg_id == 0:
        db_out = sbm.alloc_heap(shape=(co_tile, n_co), dtype=dy.dtype, name="dxps_db_out")
        for c in range(n_co):
            c0 = c * co_tile
            co_len = min(co_tile, C_out - c0)
            nisa.tensor_copy(dst=db_out[:co_len, c : c + 1], src=db_part[:co_len, c : c + 1])
            nisa.dma_copy(dst=db.slice(dim=0, start=c0, end=c0 + co_len, step=1), src=db_out[:co_len, c])
        sbm.pop_heap()  # db_out
    sbm.pop_heap()  # db_part


def _dxc_alloc(cfg, filters, sbm):
    """ALLOCATION: transposed filters Wt, and the band size (over D for 3D, H for 2D) sizing dy_res/plane/g to fit SBUF."""
    P_MAX, F_MAX = nl.tile_size.pmax, nl.tile_size.psum_fmax
    dtype, C_in, C_out = filters.dtype, cfg.C_in, cfg.C_out
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    dsz = sizeinbytes(dtype)
    co_tile, n_co, n_taps = cfg.co_tile, cfg.n_co, cfg.n_taps
    filt_flat = filters.reshape((cfg.K_d * cfg.K_h * cfg.K_w, C_in, C_out))
    wt = _build_wt_per_tap(sbm, filt_flat, n_taps, C_in, C_out, co_tile, n_co, dtype, "dx_wt")

    # Band the outer axis (D for 3D, else H); size it so dy_res + plane + result + g fit SBUF.
    outer_is_d = D_out > 1
    s_outer = cfg.stride_d if outer_is_d else cfg.stride_h
    row_stride = H_out * W_out if outer_is_d else W_out  # dy cols per outer-output row
    budget = max(1, sbm.get_free_space() - _SBUF_HEADROOM)

    # Image packing (2D branch only): pack several batch images side-by-side on the moving free axis.
    sp_plane = H_out * W_out  # max moving free per image per matmul
    if outer_is_d:
        imgs_per_pack = 1
    else:
        imgs_this_core = (-(-cfg.B // cfg.n_prgs)) if cfg.is_sharded else cfg.B
        imgs_per_pack = max(1, min(F_MAX // max(1, sp_plane), imgs_this_core))

    def _band_bytes(band, ring_deg=2):
        g_outer = band // s_outer + 2  # outer output rows the band pulls
        band_pos = g_outer * row_stride  # dy_res / g positions
        plane_cols = (band * cfg.H * cfg.W) if outer_is_d else (band * cfg.W)
        per_img = (
            ring_deg * n_co * band_pos * dsz
            + band_pos * _ACC_DTYPE_SIZE
            + plane_cols * _ACC_DTYPE_SIZE
            + plane_cols * dsz
        )
        return imgs_per_pack * per_img

    band = cfg.D if outer_is_d else cfg.H
    while band > 1 and _band_bytes(band) > budget:
        band = max(1, band // 2)
    # Shrink pack if a single band still overflows (rare); keep at least one image.
    while imgs_per_pack > 1 and _band_bytes(band) > budget:
        imgs_per_pack -= 1
    # Ring degree: 2 lets the next band's dy load overlap the current band's matmuls.
    ring_deg = 2
    if _band_bytes(band, ring_deg=2) > budget:
        ring_deg = 1
    band_pos_img = (band // s_outer + 2) * row_stride
    plane_cols_img = (band * cfg.H * cfg.W) if outer_is_d else (band * cfg.W)
    dy_res_ring = [
        [
            sbm.alloc_heap(shape=(co_tile, imgs_per_pack * band_pos_img), dtype=dtype, name=f"dx_dyres{r}_{c}")
            for c in range(n_co)
        ]
        for r in range(ring_deg)
    ]
    g_sbuf = sbm.alloc_heap(shape=(P_MAX, imgs_per_pack * band_pos_img), dtype=_ACC_DTYPE, name="dx_g")
    plane = sbm.alloc_heap(shape=(P_MAX, imgs_per_pack * plane_cols_img), dtype=_ACC_DTYPE, name="dx_plane")
    result = sbm.alloc_heap(shape=(P_MAX, imgs_per_pack * plane_cols_img), dtype=dtype, name="dx_result")
    # Two PSUM accumulator banks. The single-image (imgs_per_pack==1) 2D scatter rotates them so
    g_psum = [
        nl.ndarray(shape=(P_MAX, F_MAX), dtype=_ACC_DTYPE, buffer=nl.psum, address=(0, off))
        for off in _psum_offsets(cfg, "dx_col2im")["dxc_g"]
    ]
    return {
        "wt": wt,
        "dy_res_ring": dy_res_ring,
        "g_sbuf": g_sbuf,
        "plane": plane,
        "result": result,
        "g_psum": g_psum,
        "band": band,
        "outer_is_d": outer_is_d,
        "row_stride": row_stride,
        "imgs_per_pack": imgs_per_pack,
        "band_pos_img": band_pos_img,
        "plane_cols_img": plane_cols_img,
        "ring_deg": ring_deg,
    }


def _dxc_load_dy_band(cfg, t, dy_flat, dy_res, pack_bs, oo0, oo1):
    """LOAD: dy for outer-output rows [oo0, oo1) into dy_res, front-packed, for each image in the pack."""
    rs = t["row_stride"]
    bpi = t["band_pos_img"]
    pos0, pos1 = oo0 * rs, oo1 * rs
    n = pos1 - pos0
    for j, b in enumerate(pack_bs):
        dyb = dy_flat.select(dim=0, index=b)  # [C_out, spatial_out]
        base = j * bpi
        for c in range(cfg.n_co):
            c0 = c * cfg.co_tile
            co_len = min(cfg.co_tile, cfg.C_out - c0)
            nisa.dma_copy(
                dst=dy_res[c][:co_len, base : base + n],
                src=dyb.slice(dim=0, start=c0, end=c0 + co_len, step=1).slice(dim=1, start=pos0, end=pos1, step=1),
            )


def _dxc_scatter_band_3d(cfg, t, dx, dy_res, b, ci0, ci_len, oo0, oo1, io0, io1):
    """3D branch: contract C_out into g per tap, overlap-add scatter onto the D-band plane, store the band's D rows."""
    F_MAX = nl.tile_size.psum_fmax
    C_out, n_co, co_tile = cfg.C_out, cfg.n_co, cfg.co_tile
    D_out, H_out, W_out = cfg.D_out, cfg.H_out, cfg.W_out
    sd, sh, sw = cfg.stride_d, cfg.stride_h, cfg.stride_w
    dld, dlh, dlw = cfg.dilation_d, cfg.dilation_h, cfg.dilation_w
    wt = t["wt"]
    g_sbuf, plane, result = t["g_sbuf"], t["plane"], t["result"]
    g_psum = t["g_psum"][0]  # 3D path uses a single accumulator bank
    taps = [(kd, kh, kw) for kd in range(cfg.K_d) for kh in range(cfg.K_h) for kw in range(cfg.K_w)]
    bd = io1 - io0
    band_cols = bd * cfg.H * cfg.W
    nisa.memset(dst=plane[:ci_len, :band_cols], value=0.0)
    plane4 = plane[:ci_len, :band_cols].reshape((ci_len, bd, cfg.H, cfg.W))
    for t_idx, (kd, kh, kw) in enumerate(taps):
        do_lo, do_hi = _valid_output_range(D_out, cfg.D, kd, sd, dld, cfg.pad_d_left)
        ho_lo, ho_hi = _valid_output_range(H_out, cfg.H, kh, sh, dlh, cfg.pad_h_top)
        wo_lo, wo_hi = _valid_output_range(W_out, cfg.W, kw, sw, dlw, cfg.pad_w_left)
        offd = kd * dld - cfg.pad_d_left
        do_lo = max(do_lo, -(-(io0 - offd) // sd))  # ceil div: first do landing >= io0
        do_hi = min(do_hi, -(-(io1 - offd) // sd))
        if do_hi <= do_lo or ho_hi <= ho_lo or wo_hi <= wo_lo:
            continue
        do_n, ho_n, wo_n = do_hi - do_lo, ho_hi - ho_lo, wo_hi - wo_lo
        # Gather the tap's (do,ho,wo) box into g, front-packed [do_n, ho_n, wo_n].
        hstep = max(1, F_MAX // wo_n)
        gi = 0
        for dd in range(do_lo, do_hi):
            for hb in range(ho_lo, ho_hi, hstep):
                hn = min(hstep, ho_hi - hb)
                mv = hn * wo_n
                d_off = (dd - oo0) * H_out * W_out + hb * W_out + wo_lo  # oo-relative into dy_res
                for c in range(n_co):
                    co_len = min(co_tile, C_out - c * co_tile)
                    band_pos = dy_res[c].shape[1]
                    moving = dy_res[c].ap(
                        pattern=[[band_pos, co_len], [W_out, hn], [1, wo_n]],
                        offset=d_off,
                    )
                    nisa.nc_matmul(
                        dst=g_psum[:ci_len, :mv],
                        stationary=wt[t_idx][c][:co_len, ci0 : ci0 + ci_len],
                        moving=moving,
                        accumulate=(c > 0),
                    )
                nisa.tensor_copy(dst=g_sbuf[:ci_len, gi : gi + mv], src=g_psum[:ci_len, :mv])
                gi += mv
        g4 = g_sbuf[:ci_len, : do_n * ho_n * wo_n].reshape((ci_len, do_n, ho_n, wo_n))
        in_d = do_lo * sd + offd - io0
        in_h = ho_lo * sh + kh * dlh - cfg.pad_h_top
        in_w = wo_lo * sw + kw * dlw - cfg.pad_w_left
        dst = (
            plane4.slice(dim=1, start=in_d, end=in_d + (do_n - 1) * sd + 1, step=sd)
            .slice(dim=2, start=in_h, end=in_h + (ho_n - 1) * sh + 1, step=sh)
            .slice(dim=3, start=in_w, end=in_w + (wo_n - 1) * sw + 1, step=sw)
        )
        nisa.tensor_tensor(dst=dst, data1=dst, data2=g4, op=nl.add)
    nisa.tensor_copy(dst=result[:ci_len, :band_cols], src=plane[:ci_len, :band_cols])
    dst_hbm = (
        dx.select(dim=0, index=b)
        .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
        .slice(dim=1, start=io0, end=io1, step=1)
        .flatten_dims(1, 3)
    )
    nisa.dma_copy(dst=dst_hbm, src=result[:ci_len, :band_cols])


def _dxc_scatter_band_2d(cfg, t, dx, dy_res, pack_bs, ci0, ci_len, oo0, oo1, io0, io1):
    """2D scatter dispatcher."""
    if len(pack_bs) == 1:
        _dxc_scatter_band_2d_rowpack(cfg, t, dx, dy_res, pack_bs[0], ci0, ci_len, oo0, oo1, io0, io1)
    else:
        _dxc_scatter_band_2d_imgpack(cfg, t, dx, dy_res, pack_bs, ci0, ci_len, oo0, oo1, io0, io1)


def _dxc_scatter_band_2d_imgpack(cfg, t, dx, dy_res, pack_bs, ci0, ci_len, oo0, oo1, io0, io1):
    """Image-packed 2D scatter: stack pack_n batch images side-by-side on the moving free axis."""
    F_MAX = nl.tile_size.psum_fmax
    C_out, n_co, co_tile = cfg.C_out, cfg.n_co, cfg.co_tile
    H_out, W_out = cfg.H_out, cfg.W_out
    sh, sw = cfg.stride_h, cfg.stride_w
    dlh, dlw = cfg.dilation_h, cfg.dilation_w
    wt = t["wt"]
    g_sbuf, plane, result = t["g_sbuf"], t["plane"], t["result"]
    g_psum = t["g_psum"][0]
    bpi = t["band_pos_img"]
    pack_n = len(pack_bs)
    taps = [(kh, kw) for kh in range(cfg.K_h) for kw in range(cfg.K_w)]
    bih = io1 - io0
    band_cols = bih * cfg.W
    used_plane = pack_n * band_cols
    nisa.memset(dst=plane[:ci_len, :used_plane], value=0.0)
    plane4 = plane[:ci_len, :used_plane].reshape((ci_len, pack_n, bih, cfg.W))
    for t_idx, (kh, kw) in enumerate(taps):
        ho_lo, ho_hi = _valid_output_range(H_out, cfg.H, kh, sh, dlh, cfg.pad_h_top)
        wo_lo, wo_hi = _valid_output_range(W_out, cfg.W, kw, sw, dlw, cfg.pad_w_left)
        offh = kh * dlh - cfg.pad_h_top
        ho_lo = max(ho_lo, -(-(io0 - offh) // sh))
        ho_hi = min(ho_hi, -(-(io1 - offh) // sh))
        if ho_hi <= ho_lo or wo_hi <= wo_lo:
            continue
        ho_n, wo_n = ho_hi - ho_lo, wo_hi - wo_lo
        g4 = g_sbuf[:ci_len, : pack_n * ho_n * wo_n].reshape((ci_len, pack_n, ho_n, wo_n))
        # Pack pack_n images plus as many ho rows as fit into one wide matmul so a single
        hstep = max(1, F_MAX // (pack_n * wo_n))
        for hb in range(ho_lo, ho_hi, hstep):
            hn = min(hstep, ho_hi - hb)
            mv = pack_n * hn * wo_n
            h_off = (hb - oo0) * W_out + wo_lo  # oo-relative into dy_res (per-image)
            for c in range(n_co):
                co_len = min(co_tile, C_out - c * co_tile)
                ps = dy_res[c].shape[1]
                moving = dy_res[c].ap(
                    pattern=[[ps, co_len], [bpi, pack_n], [W_out, hn], [1, wo_n]],
                    offset=h_off,
                )
                nisa.nc_matmul(
                    dst=g_psum[:ci_len, :mv],
                    stationary=wt[t_idx][c][:co_len, ci0 : ci0 + ci_len],
                    moving=moving,
                    accumulate=(c > 0),
                )
            hb_rel = hb - ho_lo
            gpsum4 = g_psum[:ci_len, :mv].reshape((ci_len, pack_n, hn, wo_n))
            nisa.tensor_copy(dst=g4[:, :, hb_rel : hb_rel + hn, :], src=gpsum4)
        in_h = ho_lo * sh + offh - io0
        in_w = wo_lo * sw + kw * dlw - cfg.pad_w_left
        dst = plane4.slice(dim=2, start=in_h, end=in_h + (ho_n - 1) * sh + 1, step=sh).slice(
            dim=3, start=in_w, end=in_w + (wo_n - 1) * sw + 1, step=sw
        )
        nisa.tensor_tensor(dst=dst, data1=dst, data2=g4, op=nl.add)
    nisa.tensor_copy(dst=result[:ci_len, :used_plane], src=plane[:ci_len, :used_plane])
    result3 = result[:ci_len, :used_plane].reshape((ci_len, pack_n, band_cols))
    for j, b in enumerate(pack_bs):
        dst_hbm = (
            dx.select(dim=0, index=b)
            .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
            .select(dim=1, index=0)
            .slice(dim=1, start=io0, end=io1, step=1)
            .flatten_dims(1, 2)
        )
        nisa.dma_copy(dst=dst_hbm, src=result3[:, j, :])


def _dxc_scatter_band_2d_rowpack(cfg, t, dx, dy_res, b, ci0, ci_len, oo0, oo1, io0, io1):
    """Single-image 2D scatter: row-pack narrow outputs and width-tile wide outputs."""
    F_MAX = nl.tile_size.psum_fmax
    C_out, n_co, co_tile = cfg.C_out, cfg.n_co, cfg.co_tile
    H_out, W_out = cfg.H_out, cfg.W_out
    sh, sw = cfg.stride_h, cfg.stride_w
    dlh, dlw = cfg.dilation_h, cfg.dilation_w
    wt = t["wt"]
    plane, result, g_psum = t["plane"], t["result"], t["g_psum"]
    n_banks = len(g_psum)
    taps = [(kh, kw) for kh in range(cfg.K_h) for kw in range(cfg.K_w)]
    bih = io1 - io0
    band_cols = bih * cfg.W
    nisa.memset(dst=plane[:ci_len, :band_cols], value=0.0, engine=nisa.gpsimd_engine)
    plane3 = plane[:ci_len, :band_cols].reshape((ci_len, bih, cfg.W))
    rows_per_chunk = max(1, F_MAX // W_out)
    bank_i = 0
    for t_idx, (kh, kw) in enumerate(taps):
        ho_lo, ho_hi = _valid_output_range(H_out, cfg.H, kh, sh, dlh, cfg.pad_h_top)
        wo_lo, wo_hi = _valid_output_range(W_out, cfg.W, kw, sw, dlw, cfg.pad_w_left)
        offh = kh * dlh - cfg.pad_h_top
        ho_lo = max(ho_lo, -(-(io0 - offh) // sh))
        ho_hi = min(ho_hi, -(-(io1 - offh) // sh))
        if ho_hi <= ho_lo or wo_hi <= wo_lo:
            continue
        ho_n, wo_n = ho_hi - ho_lo, wo_hi - wo_lo
        base = (ho_lo - oo0) * W_out  # oo-relative col of the first packed row
        in_h0 = ho_lo * sh + offh - io0
        in_w = wo_lo * sw + kw * dlw - cfg.pad_w_left
        if W_out > F_MAX:
            for ho in range(ho_lo, ho_hi):
                in_h = ho * sh + offh - io0
                for wo0 in range(wo_lo, wo_hi, F_MAX):
                    wo1 = min(wo0 + F_MAX, wo_hi)
                    width = wo1 - wo0
                    cb = (ho - oo0) * W_out + wo0
                    gp = g_psum[bank_i]
                    bank_i = (bank_i + 1) % n_banks
                    for co_tile_idx in range(n_co):
                        co_len = min(co_tile, C_out - co_tile_idx * co_tile)
                        nisa.nc_matmul(
                            dst=gp[:ci_len, :width],
                            stationary=wt[t_idx][co_tile_idx][:co_len, ci0 : ci0 + ci_len],
                            moving=dy_res[co_tile_idx][:co_len, cb : cb + width],
                            accumulate=(co_tile_idx > 0),
                        )
                    src = gp[:ci_len, :width].reshape((ci_len, 1, width))
                    chunk_in_w = wo0 * sw + kw * dlw - cfg.pad_w_left
                    dst = plane3.slice(dim=1, start=in_h, end=in_h + 1, step=1).slice(
                        dim=2,
                        start=chunk_in_w,
                        end=chunk_in_w + (width - 1) * sw + 1,
                        step=sw,
                    )
                    nisa.tensor_tensor(dst=dst, data1=dst, data2=src, op=nl.add)
            continue
        for r0 in range(0, ho_n, rows_per_chunk):
            rc = min(rows_per_chunk, ho_n - r0)
            span = rc * W_out
            cb = base + r0 * W_out
            gp = g_psum[bank_i]
            bank_i = (bank_i + 1) % n_banks
            for c in range(n_co):
                co_len = min(co_tile, C_out - c * co_tile)
                nisa.nc_matmul(
                    dst=gp[:ci_len, :span],
                    stationary=wt[t_idx][c][:co_len, ci0 : ci0 + ci_len],
                    moving=dy_res[c][:co_len, cb : cb + span],
                    accumulate=(c > 0),
                )
            src = gp[:ci_len, :span].reshape((ci_len, rc, W_out)).slice(dim=2, start=wo_lo, end=wo_hi, step=1)
            chunk_in_h = in_h0 + r0 * sh
            dst = plane3.slice(dim=1, start=chunk_in_h, end=chunk_in_h + (rc - 1) * sh + 1, step=sh).slice(
                dim=2, start=in_w, end=in_w + (wo_n - 1) * sw + 1, step=sw
            )
            nisa.tensor_tensor(dst=dst, data1=dst, data2=src, op=nl.add)
    nisa.tensor_copy(dst=result[:ci_len, :band_cols], src=plane[:ci_len, :band_cols], engine=nisa.scalar_engine)
    dst_hbm = (
        dx.select(dim=0, index=b)
        .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
        .select(dim=1, index=0)
        .slice(dim=1, start=io0, end=io1, step=1)
        .flatten_dims(1, 2)
    )
    nisa.dma_copy(dst=dst_hbm, src=result[:ci_len, :band_cols])


def _dxc_free(sbm, cfg, ring_deg=2):
    sbm.pop_heap()  # result
    sbm.pop_heap()  # plane
    sbm.pop_heap()  # g_sbuf
    for _ in range(ring_deg * cfg.n_co):
        sbm.pop_heap()  # dy_res ring
    for _ in range(cfg.n_taps):
        for _ in range(cfg.n_co):
            sbm.pop_heap()  # wt


def _conv3d_dx_col2im(cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager) -> nl.NkiTensor:
    """General dx: per-tap C_out contraction + overlap-add scatter, banded over the outer axis to fit SBUF. Batch-sharded."""
    dx = nl.ndarray(shape=(cfg.B, cfg.C_in, cfg.D, cfg.H, cfg.W), dtype=filters.dtype, buffer=nl.shared_hbm)
    dy_flat = dy.flatten_dims(2, 4)  # [B, C_out, spatial_out]
    t = _dxc_alloc(cfg, filters, sbm)
    outer_is_d = t["outer_is_d"]
    band = t["band"]
    if outer_is_d:
        n_outer, s, dl, pad, out_n, k = cfg.D, cfg.stride_d, cfg.dilation_d, cfg.pad_d_left, cfg.D_out, cfg.K_d
    else:
        n_outer, s, dl, pad, out_n, k = cfg.H, cfg.stride_h, cfg.dilation_h, cfg.pad_h_top, cfg.H_out, cfg.K_h
    my_b = list(range(cfg.prg_id, cfg.B, cfg.n_prgs)) if cfg.is_sharded else list(range(cfg.B))
    ipp = t["imgs_per_pack"]
    dy_res_ring = t["dy_res_ring"]
    ring_deg = t["ring_deg"]
    # Build the flat band schedule so the next band's dy load can be hoisted ahead of its matmuls.
    schedule = []
    for p0 in range(0, len(my_b), ipp):
        pack_bs = my_b[p0 : p0 + ipp]
        for io0 in range(0, n_outer, band):
            io1 = min(n_outer, io0 + band)
            oo0 = max(0, -(-(io0 - (k - 1) * dl + pad) // s))
            oo1 = min(out_n, -(-(io1 + pad) // s))
            if oo1 <= oo0:
                continue
            schedule.append((pack_bs, oo0, oo1, io0, io1))
    if schedule and ring_deg == 2:
        pb0, o0, o1, _, _ = schedule[0]
        _dxc_load_dy_band(cfg, t, dy_flat, dy_res_ring[0], pb0, o0, o1)
    for idx, (pack_bs, oo0, oo1, io0, io1) in enumerate(schedule):
        cur = idx % ring_deg
        if ring_deg == 2:
            if idx + 1 < len(schedule):  # prefetch next band into the alternate ring buffer
                nb, no0, no1, _, _ = schedule[idx + 1]
                _dxc_load_dy_band(cfg, t, dy_flat, dy_res_ring[1 - cur], nb, no0, no1)
        else:
            _dxc_load_dy_band(cfg, t, dy_flat, dy_res_ring[0], pack_bs, oo0, oo1)
        cur_dy = dy_res_ring[cur]
        for i in range(cfg.n_ci):
            ci0 = i * cfg.ci_tile
            ci_len = min(cfg.ci_tile, cfg.C_in - ci0)
            if outer_is_d:
                for b in pack_bs:
                    _dxc_scatter_band_3d(cfg, t, dx, cur_dy, b, ci0, ci_len, oo0, oo1, io0, io1)
            else:
                _dxc_scatter_band_2d(cfg, t, dx, cur_dy, pack_bs, ci0, ci_len, oo0, oo1, io0, io1)
    _dxc_free(sbm, cfg, ring_deg=ring_deg)
    return dx


def _sw_db_init(ctx):
    """Allocate the running batch-local db accumulator [co, n_co] and zero it."""
    ctx.db_part = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)
    ctx.db_partial = nl.ndarray(shape=(ctx.P_MAX, 1), dtype=_ACC_DTYPE, buffer=nl.sbuf)
    for co_t in range(ctx.n_co):
        co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
        nisa.memset(dst=ctx.db_part[:co_size, co_t : co_t + 1], value=0.0)


def _sw_db_accum(ctx, cur):
    """Add the current batch's dy (already resident in dy_sets[cur]) into the running db."""
    dy_res = ctx.dy_sets[cur]
    for co_t in range(ctx.n_co):
        co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
        nisa.tensor_reduce(
            dst=ctx.db_partial[:co_size, :],
            op=nl.add,
            data=dy_res[co_t][:co_size, : ctx.P_out],
            axis=1,
            keepdims=True,
        )
        nisa.tensor_tensor(
            dst=ctx.db_part[:co_size, co_t : co_t + 1],
            data1=ctx.db_part[:co_size, co_t : co_t + 1],
            data2=ctx.db_partial[:co_size, :],
            op=nl.add,
        )


def _sw_db_write(ctx, db_recv):
    db_out = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=ctx.dy.dtype, buffer=nl.sbuf)
    for co_t in range(ctx.n_co):
        c0 = co_t * ctx.co_tile
        co_size = min(ctx.co_tile, ctx.C_out - c0)
        if db_recv != None:
            red = nl.ndarray(shape=(ctx.P_MAX, 1), dtype=_ACC_DTYPE, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=red[:co_size, :],
                data1=ctx.db_part[:co_size, co_t : co_t + 1],
                data2=db_recv[:co_size, co_t : co_t + 1],
                op=nl.add,
            )
            nisa.tensor_copy(dst=db_out[:co_size, co_t : co_t + 1], src=red[:co_size, :])
        else:
            nisa.tensor_copy(dst=db_out[:co_size, co_t : co_t + 1], src=ctx.db_part[:co_size, co_t : co_t + 1])
        nisa.dma_copy(
            dst=ctx.db.slice(dim=0, start=c0, end=c0 + co_size, step=1),
            src=db_out[:co_size, co_t],
            dge_mode=nisa.dge_mode.none,
        )


def _sw_db_finish(ctx):
    """Recombine batch-local db partials."""
    if ctx.reduce_across:
        other = 1 - ctx.prg_id
        db_recv = nl.ndarray(shape=(ctx.P_MAX, ctx.n_co), dtype=_ACC_DTYPE, buffer=nl.sbuf)
        nisa.sendrecv(src=ctx.db_part, dst=db_recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        if ctx.prg_id == 0:
            _sw_db_write(ctx, db_recv)
    elif ctx.prg_id == 0:
        _sw_db_write(ctx, None)
