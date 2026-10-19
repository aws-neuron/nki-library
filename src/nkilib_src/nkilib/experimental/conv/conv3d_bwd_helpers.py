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

"""conv3d_bwd building blocks: constants, geometry, enums, config/context state, and the alloc/load/store/transpose/reduce ops the compute kernels are assembled from."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil

_ACC_DTYPE = nl.float32
_ACC_DTYPE_SIZE = sizeinbytes(_ACC_DTYPE)
_NUM_PSUM_BANKS = 8
_PSUM_BANK_FMAX = 512
_SBUF_HEADROOM = 8192
_SBUF_ALIGN = 32  # Align transposed free-dimension tiles for SBUF DMA access.


def _forward_out_dim(in_size: int, k: int, stride: int, pad_lo: int, pad_hi: int, dilation: int) -> int:
    """Forward output extent along one spatial axis."""
    return (in_size + pad_lo + pad_hi - dilation * (k - 1) - 1) // stride + 1


def _decode_spatial(idx: int, D_out: int, H_out: int, W_out: int) -> tuple[int, int, int]:
    """Decode a flattened output-spatial index into (do, ho, wo)."""
    wo = idx % W_out
    ho = (idx // W_out) % H_out
    do = idx // (W_out * H_out)
    return do, ho, wo


def _valid_output_range(
    out_size: int, in_size: int, k: int, stride: int, dilation: int, pad_lo: int
) -> tuple[int, int]:
    """Contiguous [o_lo, o_hi) of output positions whose tap-shifted input index is in bounds."""
    offset = k * dilation - pad_lo
    o_lo = 0 if offset >= 0 else (-offset + stride - 1) // stride
    o_hi_inc = (in_size - 1 - offset) // stride
    o_hi = min(out_size, o_hi_inc + 1)
    o_lo = max(0, o_lo)
    if o_hi <= o_lo:
        return 0, 0
    return o_lo, o_hi


def _tap_valid_ranges(cfg: "Conv3dBwdConfig", kd: int, kh: int, kw: int) -> tuple[int, int, int, int, int, int]:
    """Per-tap valid (do,ho,wo) output ranges for a stride-1-dilation dw tap."""
    do_lo, do_hi = _valid_output_range(cfg.D_out, cfg.D, kd, cfg.stride_d, 1, cfg.pad_d_left)
    ho_lo, ho_hi = _valid_output_range(cfg.H_out, cfg.H, kh, cfg.stride_h, 1, cfg.pad_h_top)
    wo_lo, wo_hi = _valid_output_range(cfg.W_out, cfg.W, kw, cfg.stride_w, 1, cfg.pad_w_left)
    return do_lo, do_hi, ho_lo, ho_hi, wo_lo, wo_hi


def _dw_hbm_view(
    dw: nl.NkiTensor, kd: int, kh: int, kw: int, ci_start: int, ci_end: int, co_start: int, co_end: int
) -> nl.NkiTensor:
    """The dw[kd, kh, kw, ci_range, co_range] destination slice."""
    return (
        dw.select(dim=0, index=kd)
        .select(dim=0, index=kh)
        .select(dim=0, index=kw)
        .slice(dim=0, start=ci_start, end=ci_end, step=1)
        .slice(dim=1, start=co_start, end=co_end, step=1)
    )


def _cross_core_reduce_dw(ctx: "Conv3dBwdCtx", dw_part: nl.NkiTensor, tile_col: Callable[[int, int], int]) -> None:
    """Tier-0 dw cross-core reduce+store (2-rank shard)."""
    other = 1 - ctx.prg_id
    dw_recv = nl.ndarray(shape=dw_part.shape, dtype=_ACC_DTYPE, buffer=nl.sbuf)
    nisa.sendrecv(src=dw_part, dst=dw_recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
    rper = div_ceil(ctx.n_ci, ctx.n_prgs)
    for ci_t in range(min(rper * ctx.prg_id, ctx.n_ci), min(rper * ctx.prg_id + rper, ctx.n_ci)):
        c0i, c1i = ci_t * ctx.ci_tile, min((ci_t + 1) * ctx.ci_tile, ctx.C_in)
        ci_size = c1i - c0i
        for co_t in range(ctx.n_co):
            c0, c1 = co_t * ctx.co_tile, min((co_t + 1) * ctx.co_tile, ctx.C_out)
            col = tile_col(ci_t, co_t)
            red = nl.ndarray(shape=(ci_size, c1 - c0), dtype=_ACC_DTYPE, buffer=nl.sbuf)
            out = nl.ndarray(shape=(ci_size, c1 - c0), dtype=ctx.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=red,
                data1=dw_part[:ci_size, col : col + (c1 - c0)],
                data2=dw_recv[:ci_size, col : col + (c1 - c0)],
                op=nl.add,
            )
            nisa.tensor_copy(dst=out, src=red)
            nisa.dma_copy(dst=_dw_hbm_view(ctx.dw, 0, 0, 0, c0i, c1i, c0, c1), src=out, dge_mode=nisa.dge_mode.none)


def _store_via_result(
    dst_hbm: nl.NkiTensor,
    src: nl.NkiTensor,
    res: nl.NkiTensor,
    engine: object | None = None,
) -> None:
    """Tier-0 dw/dx store: stage src accumulator slice into SBUF result, then DMA to HBM."""
    if engine == None:
        nisa.tensor_copy(dst=res, src=src)
    else:
        nisa.tensor_copy(dst=res, src=src, engine=engine)
    nisa.dma_copy(dst=dst_hbm, src=res)


def _build_wt_per_tap(
    sbm: SbufManager,
    filt_flat: nl.NkiTensor,
    n_taps: int,
    C_in: int,
    C_out: int,
    co_tile: int,
    n_co: int,
    dtype: object,
    name: str,
) -> list[list[nl.NkiTensor]]:
    """Tier-1 dx filter transpose: allocate and fill the per-(tap, co-tile) transposed-filter grid."""
    wt = [
        [
            sbm.alloc_heap(shape=(co_tile, C_in), dtype=dtype, name=f"{name}{t}_{c}", align=_SBUF_ALIGN)
            for c in range(n_co)
        ]
        for t in range(n_taps)
    ]
    for t in range(n_taps):
        ftap = filt_flat.select(dim=0, index=t)  # [C_in, C_out]
        for c in range(n_co):
            co0 = c * co_tile
            co_len = min(co_tile, C_out - co0)
            nisa.dma_transpose(dst=wt[t][c][:co_len, :C_in], src=ftap.slice(dim=1, start=co0, end=co0 + co_len, step=1))
    return wt


def _pw_dx_store(
    ctx: "Conv3dBwdCtx",
    dx_psum: nl.NkiTensor,
    b: int,
    c0i: int,
    ci_size: int,
    p_off: int,
    p_size: int,
) -> None:
    """Tier-1 pointwise-dx store: drain the dx PSUM tile to SBUF and DMA it to HBM."""
    dx_sbuf = nl.ndarray(shape=(ci_size, p_size), dtype=ctx.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dx_sbuf, src=dx_psum)
    nisa.dma_copy(
        dst=ctx.dx_flat.select(dim=0, index=b)
        .slice(dim=0, start=c0i, end=c0i + ci_size, step=1)
        .slice(dim=1, start=p_off, end=p_off + p_size, step=1),
        src=dx_sbuf,
        dge_mode=nisa.dge_mode.none,
    )


def _pw_dx_contract(
    ctx: "Conv3dBwdCtx",
    dy_slot: list[nl.NkiTensor],
    dy_col0: int,
    c0i: int,
    ci_size: int,
    p_off: int,
    p_size: int,
) -> nl.NkiTensor:
    """Tier-1 pointwise-dx tile: dx[ci, p] = sum_co filtersT[co]^T @ dy[co, p] for one (ci-tile, pos-range)."""
    dx_psum = nl.ndarray(shape=(ci_size, p_size), dtype=_ACC_DTYPE, buffer=nl.psum)
    for co_t in range(ctx.n_co):
        co_size = min(ctx.co_tile, ctx.C_out - co_t * ctx.co_tile)
        nisa.nc_matmul(
            dst=dx_psum,
            stationary=ctx.filtersT[co_t][:co_size, c0i : c0i + ci_size],
            moving=dy_slot[co_t][:co_size, dy_col0 + p_off : dy_col0 + p_off + p_size],
        )
    return dx_psum


def _pe_transpose(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
    sc: nl.NkiTensor,
    identity: nl.NkiTensor,
    k: int,
    m: int,
    engine: object | None = None,
) -> None:
    """Tier-0 PE transpose: channel-major src[k, m] -> position-major dst[m, k] via identity matmul."""
    nisa.nc_matmul(dst=sc[:m, :k], stationary=src, moving=identity[:k, :k], accumulate=False)
    if engine == None:
        nisa.tensor_copy(dst=dst, src=sc[:m, :k])
    else:
        nisa.tensor_copy(dst=dst, src=sc[:m, :k], engine=engine)


class BwdStrategy(Enum):
    """How the general backward orchestrator shares the loaded dy, chosen by _plan_strategy."""

    SHARE_DY_POINTWISE = 0
    SHARE_DY_STRIDED = 1
    PER_GRADIENT = 2


class DwXLoadMethod(Enum):
    """How the dw x_in im2col tile is brought into positions-major layout [pos, n_cols]."""

    GATHER_TRANSPOSE = 0
    PHASE_SPLIT = 1
    BATCHED_GEMM = 2
    FAST_PLANE = 3
    POINTWISE = 4


class DxMethod(Enum):
    """Which dx (input-gradient) compute component runs, chosen by _plan_dx from geometry."""

    GEMM = 0  # tap-expand GEMM (2D stride-2 small-C_in stem)
    POINTWISE = 1  # direct 1x1 stride-1 channel contraction
    COL2IM = 2  # per-tap C_out contraction + overlap-add scatter (general)
    GEMM2D = 3  # im2col-over-dy GEMM for stride-1 2D (3x3 s1); folds db
    POINTWISE_STRIDED = 4  # 1x1 stride>1: batch-packed GEMM + strided DMA-store (downsample)


class DwFastVariant(Enum):
    """Which FAST_PLANE dw implementation runs (resolved by the planner into the config)."""

    GENERAL = 0  # PE-transpose, wide-co GEMM, optional db-fold + dx->dw prefetch
    PACKED = 1  # tap-pair packed (C_in<=64) with dma-transpose + dx->dw prologue


class DwUnitVariant(Enum):
    """Which SHARE_DY_POINTWISE unit-path dw implementation runs (resolved into the config)."""

    BASELINE = 0  # resident dy, per-batch dw
    PACKED = 1  # two-phase PSUM-only accumulate, batch-packed contraction (large-C/small-plane)
    FUSED = 2  # double-buffered fused dx+dw (prefetch next batch's dy during dw matmuls)


_PSUM_LAYOUT = {
    "dw_batched": [("dw_bg_mm", 1), ("dw_bg_tp", 2)],
    "dw_phasesplit": [("dw_ps_dw", 1), ("dw_ps_tp", _NUM_PSUM_BANKS - 1)],
    "dw_fast": [("dw_fast_acc", _NUM_PSUM_BANKS - 2), ("dw_fast_tp", 2)],
    "dx_gemm": [("dx_g", 2), ("dx_tpg", 2), ("dx_dxci_tp", 2)],
    "dx_pointwise": [("dxp_mm", 2)],
    "dx_col2im": [("dxc_g", 2)],
}


def _psum_offsets(cfg: "Conv3dBwdConfig", piece: str) -> dict:
    """Byte offset per PSUM role, assigned sequentially from _PSUM_LAYOUT; asserted to fit 8 banks."""
    offsets = {}
    bank = 0
    for role, default in _PSUM_LAYOUT[piece]:
        count = cfg.degree(role, default)
        offsets[role] = [(bank + s) * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE for s in range(count)]
        bank += count
    kernel_assert(
        bank <= _NUM_PSUM_BANKS,
        f"PSUM split overflow in '{piece}': {bank} > {_NUM_PSUM_BANKS} banks",
    )
    return offsets


@dataclass(frozen=True)
class Conv3dBwdConfig:
    """Unpacked, validated problem geometry for the fused conv3d backward pass."""

    # Dimensions
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

    # Forward convolution parameters
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

    # Gradients to compute
    compute_dx: bool
    compute_dw: bool
    compute_db: bool

    # LNC sharding
    grid_ndim: int
    n_prgs: int
    prg_id: int

    # Data-type
    dtype_size: int

    # Buffering degrees, keyed by buffer role
    interleave: dict

    # Strategy — the planner resolves the COMPLETE path here; dispatch/compute only read
    dw_x_load_method: "DwXLoadMethod"
    dx_method: "DxMethod"
    strategy: "BwdStrategy"
    dw_fast_variant: "DwFastVariant"  # which FAST_PLANE dw (when dw_x_load_method==FAST_PLANE)
    dw_unit_variant: "DwUnitVariant"  # which unit-path dw (when strategy==SHARE_DY_POINTWISE)

    @property
    def is_sharded(self) -> bool:
        """True when the kernel is sharded across more than one NeuronCore."""
        return self.n_prgs > 1

    def degree(self, role: str, default: int = 1) -> int:
        """Rotating-buffer count for a buffer role (planner-set; see _plan_interleave)."""
        return self.interleave.get(role, default)

    @property
    def n_taps(self) -> int:
        """Number of filter taps, K_d * K_h * K_w."""
        return self.K_d * self.K_h * self.K_w

    @property
    def n_pos(self) -> int:
        """Contraction extent: flattened B * D_out * H_out * W_out (matmul K axis)."""
        return self.B * self.D_out * self.H_out * self.W_out

    @property
    def n_cols(self) -> int:
        """im2col free extent: K_d * K_h * K_w * C_in (matmul N axis / dw column count)."""
        return self.n_taps * self.C_in

    @property
    def pos_tile(self) -> int:
        """K-tile: positions per matmul, capped at P_MAX (partition axis)."""
        return min(nl.tile_size.pmax, self.n_pos)

    @property
    def cout_tile(self) -> int:
        """M-tile: C_out per accumulator, capped at P_MAX (output partition axis)."""
        return min(nl.tile_size.pmax, self.C_out)

    @property
    def n_cols_tile(self) -> int:
        """N-tile: im2col columns per accumulator, capped at F_MAX (free axis / one bank)."""
        return min(nl.tile_size.psum_fmax, self.n_cols)

    @property
    def stride(self) -> tuple[int, int, int]:
        """(stride_d, stride_h, stride_w)."""

        return (self.stride_d, self.stride_h, self.stride_w)

    @property
    def padding(self) -> tuple[int, int, int, int, int, int]:
        """(pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)."""
        return (
            self.pad_d_left,
            self.pad_d_right,
            self.pad_h_top,
            self.pad_h_bottom,
            self.pad_w_left,
            self.pad_w_right,
        )

    @property
    def dilation(self) -> tuple[int, int, int]:
        """(dilation_d, dilation_h, dilation_w)."""
        return (self.dilation_d, self.dilation_h, self.dilation_w)

    @property
    def spatial_in(self) -> int:
        """Flattened input spatial extent D * H * W."""
        return self.D * self.H * self.W

    @property
    def spatial_out(self) -> int:
        """Flattened output spatial extent D_out * H_out * W_out."""
        return self.D_out * self.H_out * self.W_out

    @property
    def ci_tile(self) -> int:
        """C_in per partition tile, capped at P_MAX."""
        return min(nl.tile_size.pmax, self.C_in)

    @property
    def co_tile(self) -> int:
        """C_out per partition tile, capped at P_MAX."""
        return min(nl.tile_size.pmax, self.C_out)

    @property
    def n_ci(self) -> int:
        """Number of C_in tiles."""
        return div_ceil(self.C_in, self.ci_tile)

    @property
    def n_co(self) -> int:
        """Number of C_out tiles."""
        return div_ceil(self.C_out, self.co_tile)

    @property
    def co_blk(self) -> int:
        """C_out block on the dw accumulator FREE axis (up to one PSUM bank)."""
        return min(nl.tile_size.psum_fmax, self.C_out)

    @property
    def n_cb(self) -> int:
        """Number of C_out blocks on the dw accumulator free axis."""
        return div_ceil(self.C_out, self.co_blk)

    @property
    def co_tiles_per_blk(self) -> int:
        """Number of co-tiles that fit in one co-block."""
        return max(1, self.co_blk // self.co_tile)


class Conv3dBwdCtx:
    """Mutable state for the shared-dy component families (pointwise / strided)."""

    def __init__(
        self,
        cfg: Conv3dBwdConfig,
        dy: nl.NkiTensor,
        x_in: nl.NkiTensor,
        filters: nl.NkiTensor,
    ) -> None:
        self.cfg = cfg
        self.dy, self.x_in, self.filters = dy, x_in, filters
        self.n_prgs, self.prg_id = cfg.n_prgs, cfg.prg_id
        self.dtype = filters.dtype
        self.P_MAX, self.F_MAX = nl.tile_size.pmax, nl.tile_size.psum_fmax
        self.B, self.C_in, self.C_out = cfg.B, cfg.C_in, cfg.C_out
        self.D, self.H, self.W = cfg.D, cfg.H, cfg.W
        self.D_out, self.H_out, self.W_out = cfg.D_out, cfg.H_out, cfg.W_out
        self.P = cfg.spatial_in
        self.P_out = cfg.spatial_out
        self.sd, self.sh, self.sw = cfg.stride
        self.ci_tile, self.co_tile = cfg.ci_tile, cfg.co_tile
        self.n_ci, self.n_co = cfg.n_ci, cfg.n_co
        self.co_blk, self.n_cb = cfg.co_blk, cfg.n_cb
        self.co_tiles_per_blk = cfg.co_tiles_per_blk
        self.compute_db = cfg.compute_db
        self.n_pos_tiles = div_ceil(self.P_out, self.P_MAX)
        self.degree = cfg.degree("operands", 2)  # operands held N batches deep
        self.dw_dy_via_dma = self.P >= 2048 and self.n_co > 1 and cfg.dtype_size == 2
        # FUSED unit variant (resolved by the planner) runs the double-buffered fused dx+dw path
        self.dy_double_buf = cfg.dw_unit_variant == DwUnitVariant.FUSED
        _bwd_shard(self, allow_ci_co_shard=(cfg.strategy == BwdStrategy.SHARE_DY_STRIDED))
        # Unit-path (non-fused) schedule knobs — pure reorders of the same ops:
        self.pw_db_first = self.n_co == 4 and not self.dy_double_buf
        self.pw_dx_last = self.n_co == 4 and self.reduce_across and not self.dy_double_buf
        # Coalesce the x dma_transpose across batches for large planes (or single-ci mid planes).
        self.pw_coalesce_xt = self.P >= 3136 or (self.P >= 784 and self.n_ci == 1)
        # PACKED unit dw exchanges/reduces/stores dw per co-block inside _pw_dw_pk.
        self.pw_store_dw_here = cfg.dw_unit_variant != DwUnitVariant.PACKED
        self.dy_flat = dy.flatten_dims(2, 4)
        self.x_flat = x_in.flatten_dims(2, 4)
        self.filt2d = filters.select(dim=0, index=0).select(dim=0, index=0).select(dim=0, index=0)
        self.dx = nl.ndarray(shape=(self.B, self.C_in, self.D, self.H, self.W), dtype=self.dtype, buffer=nl.shared_hbm)
        self.dx_flat = self.dx.flatten_dims(2, 4)
        self.dw = nl.ndarray(
            shape=(cfg.K_d, cfg.K_h, cfg.K_w, self.C_in, self.C_out), dtype=self.dtype, buffer=nl.shared_hbm
        )
        self.db = nl.ndarray(shape=(self.C_out,), dtype=dy.dtype, buffer=nl.shared_hbm) if cfg.compute_db else None
        self.filtersT = self.identity = self.dw_acc = self.dw_psum = None
        self.dy_res = self.db_part = self.dy_res_buf = None
        self.out_bufs = self.dx_acc = self.dx_acc_flat = None
        self.dy_sets = self.x_full_sets = self.xs_sets = None


def _bwd_shard(ctx: Conv3dBwdCtx, allow_ci_co_shard: bool = True) -> None:
    """Pick the core's shard: batch (default), or C_in / C_out when the dw accumulators overflow PSUM."""
    n_prgs, prg_id = ctx.n_prgs, ctx.prg_id
    banks_batch = ctx.n_ci * ctx.n_cb
    avail = _NUM_PSUM_BANKS - 2
    ctx.shard_ci = (
        allow_ci_co_shard
        and n_prgs > 1
        and ctx.n_ci >= n_prgs
        and banks_batch > avail
        and div_ceil(ctx.n_ci, n_prgs) * ctx.n_cb <= avail
    )
    ctx.shard_co = (
        allow_ci_co_shard
        and n_prgs > 1
        and not ctx.shard_ci
        and ctx.n_cb >= n_prgs
        and banks_batch > avail
        and ctx.n_ci * div_ceil(ctx.n_cb, n_prgs) <= avail
    )
    ctx.co_b_lo, ctx.co_b_hi = 0, ctx.n_cb
    if ctx.shard_ci or ctx.shard_co:
        ctx.b_lo, ctx.b_hi = 0, ctx.B
        ctx.ci_t_lo, ctx.ci_t_hi = 0, ctx.n_ci
        if ctx.shard_ci:
            per = div_ceil(ctx.n_ci, n_prgs)
            ctx.ci_t_lo = min(per * prg_id, ctx.n_ci)
            ctx.ci_t_hi = min(ctx.ci_t_lo + per, ctx.n_ci)
        else:
            per = div_ceil(ctx.n_cb, n_prgs)
            ctx.co_b_lo = min(per * prg_id, ctx.n_cb)
            ctx.co_b_hi = min(ctx.co_b_lo + per, ctx.n_cb)
    else:
        per = div_ceil(ctx.B, n_prgs)
        ctx.b_lo = min(per * prg_id, ctx.B)
        ctx.b_hi = min(ctx.b_lo + per, ctx.B)
        ctx.ci_t_lo, ctx.ci_t_hi = 0, ctx.n_ci
    ctx.Bp = ctx.b_hi - ctx.b_lo
    ctx.dw_has_work = ctx.Bp > 0
    ctx.reduce_across = n_prgs == 2 and not ctx.shard_ci and not ctx.shard_co


def _dw_alloc_tensors(cfg: Conv3dBwdConfig, sbm: SbufManager, dtype: object, dw_dtype: object) -> dict[str, object]:
    """Allocate general dw tiles into a dict: dy_t, x_chan, x_t, x_tt, result, plus eight PSUM banks (acc)."""
    pos_tile = cfg.pos_tile
    cout_tile = cfg.cout_tile
    n_cols_tile = cfg.n_cols_tile

    dy_t = [
        sbm.alloc_heap(shape=(pos_tile, cout_tile), dtype=dtype, name=f"dw_dy_t{i}", align=_SBUF_ALIGN)
        for i in range(cfg.degree("dy", 2))
    ]
    x_chan_cols = min(nl.tile_size.pmax, n_cols_tile)
    x_chan = [
        sbm.alloc_heap(shape=(x_chan_cols, pos_tile), dtype=dtype, name=f"dw_x_chan{i}")
        for i in range(cfg.degree("x", 2))
    ]

    x_t = [
        sbm.alloc_heap(shape=(pos_tile, n_cols_tile), dtype=dtype, name=f"dw_x_t{i}", align=_SBUF_ALIGN)
        for i in range(cfg.degree("x", 2))
    ]
    x_chan_cols_p = min(nl.tile_size.pmax, n_cols_tile)
    x_tt = [
        sbm.alloc_heap(shape=(pos_tile, x_chan_cols_p), dtype=dtype, name=f"dw_x_tt{i}", align=_SBUF_ALIGN)
        for i in range(cfg.degree("x", 2))
    ]

    acc_part, acc_free = cout_tile, n_cols_tile
    result = sbm.alloc_heap(shape=(acc_part, acc_free), dtype=dw_dtype, name="dw_result")

    acc = [
        nl.ndarray(
            shape=(acc_part, acc_free),
            dtype=_ACC_DTYPE,
            buffer=nl.psum,
            address=(0, bank * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
        )
        for bank in range(_NUM_PSUM_BANKS)
    ]
    return {"dy_t": dy_t, "x_chan": x_chan, "x_t": x_t, "x_tt": x_tt, "result": result, "acc": acc}


def _dw_free_tensors(sbm: SbufManager, tensors: dict[str, object]) -> None:
    """Pop every SBUF heap alloc from _dw_alloc_tensors (reverse order; PSUM acc is not heap)."""
    sbm.pop_heap()  # result
    for _ in tensors["x_tt"]:
        sbm.pop_heap()
    for _ in tensors["x_t"]:
        sbm.pop_heap()
    for _ in tensors["x_chan"]:
        sbm.pop_heap()
    for _ in tensors["dy_t"]:
        sbm.pop_heap()


def _dw_load_x(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    x_in: nl.NkiTensor,
    buf_idx: int,
    b: int,
    pos0: int,
    pos_len: int,
    col0: int,
    col_len: int,
) -> None:
    """Load one positions-major x im2col tile (only GATHER_TRANSPOSE reaches this body)."""
    _dw_load_x_gather(cfg, tensors, x_in, buf_idx, b, pos0, pos_len, col0, col_len)


def _dw_load_x_gather(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    x_in: nl.NkiTensor,
    buf_idx: int,
    b: int,
    pos0: int,
    pos_len: int,
    col0: int,
    col_len: int,
) -> None:
    """GATHER_TRANSPOSE (universal fallback): gather im2col channel-major into x_chan, then transpose to x_t."""
    x_chan = tensors["x_chan"][buf_idx]
    x_t = tensors["x_t"][buf_idx]
    x_tt = tensors["x_tt"][buf_idx]
    C_in = cfg.C_in
    n_taps_hw = cfg.K_h * cfg.K_w
    W_out = cfg.W_out
    chan_cap = x_chan.shape[0]
    xb = x_in.select(dim=0, index=b)  # [C_in, D, H, W]

    if C_in <= chan_cap:
        taps_per_chunk = max(1, chan_cap // C_in)
        chunk_cols = taps_per_chunk * C_in
    else:
        chunk_cols = chan_cap  # channel-fragment tiling within one tap

    for cc0 in range(0, col_len, chunk_cols):
        cc_len = min(chunk_cols, col_len - cc0)
        nisa.memset(dst=x_chan[:cc_len, :pos_len], value=0.0)

        cc = 0
        while cc < cc_len:
            gcol = col0 + cc0 + cc
            tap = gcol // C_in
            ci0 = gcol % C_in  # first channel of this fragment within the tap
            ci_n = min(C_in - ci0, cc_len - cc)  # channels of this tap in this chunk
            kw = tap % cfg.K_w
            kh = (tap // cfg.K_w) % cfg.K_h
            kd = tap // n_taps_hw
            row_base = cc  # first x_chan partition row for this fragment

            i = 0
            while i < pos_len:
                do, ho, wo = _decode_spatial(pos0 + i, cfg.D_out, cfg.H_out, W_out)
                wcount = min(W_out - wo, pos_len - i)  # contiguous wo run within this row & tile
                id_ = do * cfg.stride_d + kd * cfg.dilation_d - cfg.pad_d_left
                ih = ho * cfg.stride_h + kh * cfg.dilation_h - cfg.pad_h_top
                if 0 <= id_ < cfg.D and 0 <= ih < cfg.H:
                    iw0 = wo * cfg.stride_w + kw * cfg.dilation_w - cfg.pad_w_left
                    o_lo = 0
                    if iw0 < 0:
                        o_lo = (-iw0 + cfg.stride_w - 1) // cfg.stride_w
                    o_hi = wcount
                    max_o = (cfg.W - 1 - iw0) // cfg.stride_w + 1 if iw0 < cfg.W else 0
                    o_hi = min(o_hi, max_o)
                    if o_hi > o_lo:
                        iw_start = iw0 + o_lo * cfg.stride_w
                        n = o_hi - o_lo
                        src = (
                            xb.select(dim=1, index=id_)
                            .select(dim=1, index=ih)
                            .slice(dim=0, start=ci0, end=ci0 + ci_n, step=1)
                            .slice(dim=1, start=iw_start, end=iw_start + (n - 1) * cfg.stride_w + 1, step=cfg.stride_w)
                        )
                        nisa.dma_copy(dst=x_chan[row_base : row_base + ci_n, i + o_lo : i + o_lo + n], src=src)

                i += wcount
            cc += ci_n

        nisa.dma_transpose(dst=x_tt[:pos_len, :cc_len], src=x_chan[:cc_len, :pos_len])
        nisa.tensor_copy(dst=x_t[:pos_len, cc0 : cc0 + cc_len], src=x_tt[:pos_len, :cc_len])


def _dw_load_dy(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    dy: nl.NkiTensor,
    buf_idx: int,
    b: int,
    pos0: int,
    pos_len: int,
    co0: int,
    co_len: int,
) -> None:
    """Build one positions-major dy tile into dy_t[buf_idx] via DMA transpose of the channel-major slice."""
    dy_t = tensors["dy_t"][buf_idx]

    dyb = dy.flatten_dims(2, 4).select(dim=0, index=b)  # [C_out, spatial_out] channel-major
    src = dyb.slice(dim=0, start=co0, end=co0 + co_len, step=1).slice(
        dim=1, start=pos0, end=pos0 + pos_len, step=1
    )  # [co_len, pos_len]
    nisa.dma_transpose(dst=dy_t[:pos_len, :co_len], src=src)


def _dw_compute(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    dy_buf_idx: int,
    x_buf_idx: int,
    bank_idx: int,
    co_len: int,
    col_len: int,
    pos_len: int,
    accumulate: bool,
) -> None:
    """Accumulate acc[co_len, col_len] += dy_t[pos, co]^T @ x_t[pos, col] over one K-tile."""
    nisa.nc_matmul(
        dst=tensors["acc"][bank_idx][:co_len, :col_len],
        stationary=tensors["dy_t"][dy_buf_idx][:pos_len, :co_len],
        moving=tensors["x_t"][x_buf_idx][:pos_len, :col_len],
        accumulate=accumulate,
    )


def _dw_layout(cfg: Conv3dBwdConfig, tensors: dict[str, object], bank_idx: int, part_len: int, free_len: int) -> None:
    """Evict accumulator acc[bank_idx] [co_len, col_len] from PSUM -> SBUF result (cast)."""
    nisa.tensor_copy(dst=tensors["result"][:part_len, :free_len], src=tensors["acc"][bank_idx][:part_len, :free_len])


def _dw_store(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    dw: nl.NkiTensor,
    co0: int,
    co_len: int,
    col0: int,
    col_len: int,
) -> None:
    """Write result[co, col] into dw[K_d,K_h,K_w,C_in,C_out] (flat [n_cols, C_out]) via strided DMA."""
    dw_flat = dw.reshape((cfg.n_cols, cfg.C_out))
    dst = dw_flat.ap(pattern=[[1, co_len], [cfg.C_out, col_len]], offset=col0 * cfg.C_out + co0)
    nisa.dma_copy(dst=dst, src=tensors["result"][:co_len, :col_len])


def _co_chunks(my_co: list[int], max_tiles: int) -> list[list[int]]:
    """Group owned C_out tile indices into consecutive runs (capped at max_tiles) so the dw GEMM issues one wide matmul per run."""
    chunks = []
    run = []
    for c in sorted(my_co):
        if run and c == run[-1] + 1 and len(run) < max_tiles:
            run.append(c)
        else:
            if run:
                chunks.append(run)
            run = [c]
    if run:
        chunks.append(run)
    return chunks
