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

from ...core.utils.allocator import SbufManager, align_to, sizeinbytes
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
    LOW_CHAN = 5  # stride-1 transposed-tap Conv3D for C_out <= 64


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
    def n_cols(self) -> int:
        """im2col free extent: K_d * K_h * K_w * C_in (matmul N axis / dw column count)."""
        return self.n_taps * self.C_in

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


_DW_SPATIAL_TILE = 8 * 1024
# Below this a per-row HBM run is too small to trade for halved x bytes. Measured on trn2: folding
# wins at 1536 B (-4.1%), 768 B (-4.1%) and 384 B (-9.7%) runs, so the floor sits under the
# smallest tested win; the configs it still excludes have 16-32 B rows.
_DMA_MIN_RUN_BYTES = 256


def _dw_repl_params(channels: int) -> tuple[int, int]:
    """Return batch replication and aligned partition stride."""
    if channels <= 32:
        return 4, 32
    if channels <= 64:
        return 2, 64
    return 1, nl.tile_size.pmax


def _dw_spatial_rows(cfg: Conv3dBwdConfig, pos0: int, pos_len: int) -> list[tuple[int, int, int, int, int]]:
    """Split a flattened output tile into W-row segments."""
    rows = []
    off = 0
    while off < pos_len:
        do, ho, wo = _decode_spatial(pos0 + off, cfg.D_out, cfg.H_out, cfg.W_out)
        count = min(cfg.W_out - wo, pos_len - off)
        rows.append((off, do, ho, wo, count))
        off += count
    return rows


def _dw_row_reuse_params(cfg: Conv3dBwdConfig) -> tuple[bool, int, int]:
    """Whether one x_t per k_w can serve every k_h, plus the per-k_h block step and extra blocks.

    A k_h step moves the output position by k_h * (dilation_h / stride_h) * W_out. In x_t's
    [block, partition] addressing that is a whole number of blocks -- the partition index is
    unchanged, so the matmul stationary stays a single rectangle -- only when pmax divides W_out.
    Then x_t(k_h, k_w) at block b equals x_t(0, k_w) at block b + k_h * step, so one build per k_w
    serves all K_h taps of that column at the cost of (K_h - 1) * step extra blocks.
    """
    if cfg.D_out != 1 or cfg.K_d != 1:
        return False, 0, 0  # a shifted position would roll into the next depth slice
    if cfg.K_h <= 1 or cfg.dilation_h % cfg.stride_h != 0:
        return False, 0, 0  # nothing to share, or the shifted row index is not an integer
    if cfg.W_out % nl.tile_size.pmax != 0:
        return False, 0, 0  # the k_h step would move the partition index
    step = (cfg.dilation_h // cfg.stride_h) * (cfg.W_out // nl.tile_size.pmax)
    return True, step, (cfg.K_h - 1) * step


def _dw_fold_pad_load(cfg: Conv3dBwdConfig, dtype_size: int) -> bool:
    """Whether x can be DMA'd straight into x_padded's interior, skipping the dense x staging.

    One DMA can write a strided destination, so the x -> x_padded relayout is redundant work: the
    HBM read can land in the padded interior directly. That halves x's DMA bytes and drops the x
    buffer entirely, at the cost of splitting the HBM read into one run per row (vw elements)
    rather than one contiguous run per (batch, channel). Only worth it when a row is a decent run.
    """
    return cfg.W * dtype_size >= _DMA_MIN_RUN_BYTES


def _dw_row_reuse_rows(cfg: Conv3dBwdConfig, pos0: int, pos_len: int) -> list[tuple[int, int, int, int, int]]:
    """Row segments for one tile plus the extra output rows the k_h block offsets read.

    The extra rows continue past H_out, so they are decoded linearly rather than through
    _decode_spatial, whose modulo would wrap them back to row 0 of the next depth slice.
    """
    rows = _dw_spatial_rows(cfg, pos0, pos_len)
    total = pos_len + (cfg.K_h - 1) * (cfg.dilation_h // cfg.stride_h) * cfg.W_out
    off = pos_len
    while off < total:
        ho, wo = (pos0 + off) // cfg.W_out, (pos0 + off) % cfg.W_out
        count = min(cfg.W_out - wo, total - off)
        rows.append((off, 0, ho, wo, count))
        off += count
    return rows


def _dw_tap_groups(
    cfg: Conv3dBwdConfig, tap0: int, tap_len: int, row_reuse: bool, kh_step: int
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Group a pass's taps by the x_t build they share: (build tap, [(tap, x_t block offset)])."""
    if not row_reuse:
        return [(tap, [(tap, 0)]) for tap in range(tap0, tap0 + tap_len)]
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tap in range(tap0, tap0 + tap_len):
        kd, kh, kw = _decode_spatial(tap, cfg.K_d, cfg.K_h, cfg.K_w)
        groups.setdefault((kd, kw), []).append((tap, kh * kh_step))
    return [(members[0][0], members) for members in groups.values()]


def _dw_window_geometry(cfg: Conv3dBwdConfig, pos0: int, pos_len: int) -> tuple:
    """Return row descriptors and the union input box."""
    rows = _dw_spatial_rows(cfg, pos0, pos_len)
    d0, d1 = cfg.D, 0
    h0, h1 = cfg.H, 0
    w0, w1 = cfg.W, 0
    for _, do, ho, wo, count in rows:
        d0 = min(d0, do * cfg.stride_d - cfg.pad_d_left)
        d1 = max(d1, do * cfg.stride_d + (cfg.K_d - 1) * cfg.dilation_d - cfg.pad_d_left + 1)
        h0 = min(h0, ho * cfg.stride_h - cfg.pad_h_top)
        h1 = max(h1, ho * cfg.stride_h + (cfg.K_h - 1) * cfg.dilation_h - cfg.pad_h_top + 1)
        w0 = min(w0, wo * cfg.stride_w - cfg.pad_w_left)
        w1 = max(
            w1,
            (wo + count - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w - cfg.pad_w_left + 1,
        )
    vd0, vd1 = max(0, d0), min(cfg.D, d1)
    vh0, vh1 = max(0, h0), min(cfg.H, h1)
    vw0, vw1 = max(0, w0), min(cfg.W, w1)
    return rows, d0, d1, h0, h1, w0, w1, vd0, vd1, vh0, vh1, vw0, vw1


def _dw_heap_alloc_end(heap_addr: int, bytes_per_partition: int, align: int = 4) -> int:
    """Return the aligned post-allocation heap address."""
    return align_to(heap_addr - bytes_per_partition - (align - 1), align)


def _dw_max_window_sizes(cfg: Conv3dBwdConfig) -> tuple[int, int]:
    """Return maximum valid and padded window volumes."""
    max_valid = 1
    max_padded = 1
    for pos0 in range(0, cfg.spatial_out, _DW_SPATIAL_TILE):
        pos_len = min(_DW_SPATIAL_TILE, cfg.spatial_out - pos0)
        geometry = _dw_window_geometry(cfg, pos0, pos_len)
        _, d0, d1, h0, h1, w0, w1, vd0, vd1, vh0, vh1, vw0, vw1 = geometry
        max_valid = max(max_valid, max(0, vd1 - vd0) * max(0, vh1 - vh0) * max(0, vw1 - vw0))
        max_padded = max(max_padded, (d1 - d0) * (h1 - h0) * (w1 - w0))
    return max_valid, max_padded


def _dw_alloc_tensors(
    cfg: Conv3dBwdConfig,
    sbm: SbufManager,
    dtype: object,
    dw_dtype: object,
    work_tiles: int,
    batch_shard: bool = False,
) -> dict[str, object]:
    """Allocate fallback buffers and the largest fitting work-set ring."""
    ci_tile = min(nl.tile_size.pmax, cfg.C_in)
    co_tile = min(nl.tile_size.pmax, cfg.C_out)
    ci_repl, ci_stride = _dw_repl_params(ci_tile)
    co_repl, co_stride = _dw_repl_params(co_tile)
    batch_repl = max(ci_repl, co_repl)
    x_planes = div_ceil(batch_repl, ci_repl)
    y_planes = div_ceil(batch_repl, co_repl)
    packed_ci = ci_repl * ci_stride
    packed_co = co_repl * co_stride
    spatial_blocks = div_ceil(_DW_SPATIAL_TILE, nl.tile_size.pmax)
    row_reuse, kh_step, xt_ext_blocks = _dw_row_reuse_params(cfg)
    xt_blocks = spatial_blocks + xt_ext_blocks
    max_x, max_x_padded = _dw_max_window_sizes(cfg)
    dtype_size = sizeinbytes(dtype)
    fold_pad_load = _dw_fold_pad_load(cfg, dtype_size)

    # ci_len <= ci_tile <= ci_stride, so a band's PE column tile is never narrower than the
    # channel tile it holds (see _dw_band_kwargs).
    kernel_assert(ci_tile <= ci_stride, "dw channel tile wider than its batch band")
    heap_depth = len(sbm.heap)
    reduced = sbm.alloc_heap(shape=(ci_tile, co_tile), dtype=_ACC_DTYPE, name="dw_reduced")
    recv = sbm.alloc_heap(shape=(ci_tile, co_tile), dtype=_ACC_DTYPE, name="dw_recv") if batch_shard else None
    result = sbm.alloc_heap(shape=(ci_tile, co_tile), dtype=dw_dtype, name="dw_result")
    # Both prime operands are zero and share the partition range, so one tile backs both.
    zero = sbm.alloc_heap(shape=(nl.tile_size.pmax, _PSUM_BANK_FMAX), dtype=dtype, name="dw_zero")
    nisa.memset(dst=zero, value=0.0)

    y_bytes = _DW_SPATIAL_TILE * dtype_size
    yt_bytes = spatial_blocks * packed_co * dtype_size
    x_bytes = max_x * dtype_size
    xp_bytes = max_x_padded * dtype_size
    xt_base_bytes = spatial_blocks * packed_ci * dtype_size
    xt_bytes = xt_blocks * packed_ci * dtype_size
    x_tt_bytes = packed_ci * dtype_size
    x_row_bytes = nl.tile_size.pmax * dtype_size

    stack_addr = sbm.get_stack_curr_addr()
    heap_addr = sbm.get_heap_curr_addr()

    def _ws_end(start: int, xt_b: int) -> int:
        """Heap address after one working set whose x_t tiles are xt_b bytes per partition."""
        end = start
        for _ in range(y_planes):
            end = _dw_heap_alloc_end(end, y_bytes)
            end = _dw_heap_alloc_end(end, yt_bytes, _SBUF_ALIGN)
        for _ in range(x_planes):
            if not fold_pad_load:
                end = _dw_heap_alloc_end(end, x_bytes)
            end = _dw_heap_alloc_end(end, xp_bytes)
            end = _dw_heap_alloc_end(end, xt_b, _SBUF_ALIGN)
        end = _dw_heap_alloc_end(end, x_tt_bytes, _SBUF_ALIGN)
        return _dw_heap_alloc_end(end, x_row_bytes)

    def _fit_ws(xt_b: int) -> tuple[int, int]:
        """Deepest working-set ring that fits, and the heap address it ends at."""
        count, end = 0, heap_addr
        while count < max(1, work_tiles):
            candidate = _ws_end(end, xt_b)
            if candidate < stack_addr:
                break
            end, count = candidate, count + 1
        return max(1, count), end

    kernel_assert(
        _ws_end(heap_addr, xt_base_bytes) >= stack_addr,
        f"GATHER_TRANSPOSE 8K working set does not fit SBUF: free={sbm.get_free_space()}",
    )

    ws_count, ws_heap_end = _fit_ws(xt_bytes)
    if row_reuse and ws_count < _fit_ws(xt_base_bytes)[0]:
        # Never trade working-set ring depth, which is what keeps the DMA queues fed, for the
        # longer x_t that row reuse needs.
        row_reuse, kh_step, xt_blocks, xt_bytes = False, 0, spatial_blocks, xt_base_bytes
        ws_count, ws_heap_end = _fit_ws(xt_bytes)

    # With row reuse the ring is indexed by x_t variant, so slots beyond K_d * K_w are unusable.
    max_slots = cfg.K_d * cfg.K_w if row_reuse else cfg.n_taps
    extra_xt_per_ws = 0
    extra_heap_end = ws_heap_end
    while extra_xt_per_ws + 1 < max_slots:
        candidate = extra_heap_end
        for _ in range(ws_count * x_planes):
            candidate = _dw_heap_alloc_end(candidate, xt_bytes, _SBUF_ALIGN)
        if candidate < stack_addr:
            break
        extra_heap_end = candidate
        extra_xt_per_ws += 1
    xt_ring = 1 + extra_xt_per_ws

    working_sets = []
    for ws_idx in range(ws_count):
        y = [
            sbm.alloc_heap(shape=(packed_co, _DW_SPATIAL_TILE), dtype=dtype, name=f"dw_y{ws_idx}_{p}")
            for p in range(y_planes)
        ]
        yt = [
            sbm.alloc_heap(
                shape=(nl.tile_size.pmax, spatial_blocks * packed_co),
                dtype=dtype,
                name=f"dw_yt{ws_idx}_{p}",
                align=_SBUF_ALIGN,
            )
            for p in range(y_planes)
        ]
        x = (
            None
            if fold_pad_load
            else [
                sbm.alloc_heap(shape=(packed_ci, max_x), dtype=dtype, name=f"dw_x{ws_idx}_{p}") for p in range(x_planes)
            ]
        )
        x_padded = [
            sbm.alloc_heap(shape=(packed_ci, max_x_padded), dtype=dtype, name=f"dw_xp{ws_idx}_{p}")
            for p in range(x_planes)
        ]
        xt = [
            [
                sbm.alloc_heap(
                    shape=(nl.tile_size.pmax, xt_blocks * packed_ci),
                    dtype=dtype,
                    name=f"dw_xt{ws_idx}_{slot}_{p}",
                    align=_SBUF_ALIGN,
                )
                for p in range(x_planes)
            ]
            for slot in range(xt_ring)
        ]
        x_tt = sbm.alloc_heap(
            shape=(nl.tile_size.pmax, packed_ci),
            dtype=dtype,
            name=f"dw_xtt{ws_idx}",
            align=_SBUF_ALIGN,
        )
        x_row = sbm.alloc_heap(
            shape=(packed_ci, nl.tile_size.pmax),
            dtype=dtype,
            name=f"dw_xrow{ws_idx}",
        )
        working_sets.append({"y": y, "yt": yt, "x": x, "x_padded": x_padded, "xt": xt, "x_tt": x_tt, "x_row": x_row})

    return {
        "working_sets": working_sets,
        "reduced": reduced,
        "recv": recv,
        "result": result,
        "zero": zero,
        "ci_repl": ci_repl,
        "ci_stride": ci_stride,
        "co_repl": co_repl,
        "co_stride": co_stride,
        "batch_repl": batch_repl,
        "x_planes": x_planes,
        "y_planes": y_planes,
        "packed_ci": packed_ci,
        "packed_co": packed_co,
        "spatial_blocks": spatial_blocks,
        "xt_blocks": xt_blocks,
        "row_reuse": row_reuse,
        "kh_step": kh_step,
        "fold_pad_load": fold_pad_load,
        "xt_ring": xt_ring,
        "heap_allocs": len(sbm.heap) - heap_depth,
    }


def _dw_free_tensors(sbm: SbufManager, tensors: dict[str, object]) -> None:
    """Pop fallback SBUF allocations."""
    for _ in range(tensors["heap_allocs"]):
        sbm.pop_heap()


def _dw_transpose_row(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
    pos_off: int,
    count: int,
    packed_width: int,
    partial_tmp: nl.NkiTensor | None = None,
    row_tmp: nl.NkiTensor | None = None,
    contiguous: bool = True,
) -> None:
    """Transpose one W row into block-major layout."""
    src_off = 0
    block = pos_off // nl.tile_size.pmax
    lane = pos_off % nl.tile_size.pmax
    if not contiguous:
        kernel_assert(partial_tmp != None and row_tmp != None, "strided transpose row requires scratch")
        while count > 0:
            chunk = min(nl.tile_size.pmax - lane, count)
            nisa.dma_copy(dst=row_tmp[:packed_width, :chunk], src=src[:, src_off : src_off + chunk])
            nisa.dma_transpose(dst=partial_tmp[:chunk, :packed_width], src=row_tmp[:packed_width, :chunk])
            nisa.dma_copy(
                dst=dst[lane : lane + chunk, block * packed_width : (block + 1) * packed_width],
                src=partial_tmp[:chunk, :packed_width],
            )
            src_off += chunk
            count -= chunk
            block += 1
            lane = 0
        return
    if lane != 0:
        head = min(nl.tile_size.pmax - lane, count)
        kernel_assert(partial_tmp != None, "non-aligned transpose row requires scratch")
        nisa.dma_transpose(dst=partial_tmp[:head, :packed_width], src=src[:, :head])
        nisa.dma_copy(
            dst=dst[lane : lane + head, block * packed_width : (block + 1) * packed_width],
            src=partial_tmp[:head, :packed_width],
        )
        src_off += head
        count -= head
        block += 1
    full_blocks = count // nl.tile_size.pmax
    if full_blocks > 0:
        full_len = full_blocks * nl.tile_size.pmax
        nisa.dma_transpose(
            dst=dst[
                : nl.tile_size.pmax,
                block * packed_width : (block + full_blocks) * packed_width,
            ].reshape((nl.tile_size.pmax, full_blocks, packed_width)),
            src=src[:, src_off : src_off + full_len].reshape((packed_width, full_blocks, nl.tile_size.pmax)),
        )
        src_off += full_len
        count -= full_len
        block += full_blocks
    if count > 0:
        nisa.dma_transpose(
            dst=dst[:count, block * packed_width : (block + 1) * packed_width],
            src=src[:, src_off : src_off + count],
        )


def _dw_load_work_set(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    ws_idx: int,
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    batch0: int,
    batch_len: int,
    pos0: int,
    pos_len: int,
    ci0: int,
    ci_len: int,
    co0: int,
    co_len: int,
) -> None:
    """Load dy and build the padded input window."""
    ws = tensors["working_sets"][ws_idx]
    ci_repl, ci_stride = tensors["ci_repl"], tensors["ci_stride"]
    co_repl, co_stride = tensors["co_repl"], tensors["co_stride"]
    packed_ci, packed_co = tensors["packed_ci"], tensors["packed_co"]

    dy_flat = dy.flatten_dims(2, 4)
    for j in range(batch_len):
        plane, slot = j // co_repl, j % co_repl
        nisa.dma_copy(
            dst=ws["y"][plane][slot * co_stride : slot * co_stride + co_len, :pos_len],
            src=dy_flat.select(dim=0, index=batch0 + j)
            .slice(dim=0, start=co0, end=co0 + co_len, step=1)
            .slice(dim=1, start=pos0, end=pos0 + pos_len, step=1),
        )
    for plane in range(div_ceil(batch_len, co_repl)):
        _dw_transpose_row(ws["yt"][plane], ws["y"][plane][:packed_co, :pos_len], 0, pos_len, packed_co)

    geometry = _dw_window_geometry(cfg, pos0, pos_len)
    _, d0, d1, h0, h1, w0, w1, vd0, vd1, vh0, vh1, vw0, vw1 = geometry
    pd, ph, pw = d1 - d0, h1 - h0, w1 - w0
    vd, vh, vw = max(0, vd1 - vd0), max(0, vh1 - vh0), max(0, vw1 - vw0)
    valid_volume = vd * vh * vw
    padded_volume = pd * ph * pw
    active_x_planes = div_ceil(batch_len, ci_repl)

    fold = tensors["fold_pad_load"]

    def _x_src(j: int) -> nl.NkiTensor:
        """This batch's valid input window in HBM."""
        return (
            x_in.select(dim=0, index=batch0 + j)
            .slice(dim=0, start=ci0, end=ci0 + ci_len, step=1)
            .slice(dim=1, start=vd0, end=vd1, step=1)
            .slice(dim=2, start=vh0, end=vh1, step=1)
            .slice(dim=3, start=vw0, end=vw1, step=1)
        )

    if valid_volume > 0 and not fold:
        for j in range(batch_len):
            plane, slot = j // ci_repl, j % ci_repl
            nisa.dma_copy(
                dst=ws["x"][plane][slot * ci_stride : slot * ci_stride + ci_len, :valid_volume].reshape(
                    (ci_len, vd, vh, vw)
                ),
                src=_x_src(j),
            )

    rd0, rh0, rw0 = vd0 - d0, vh0 - h0, vw0 - w0
    for plane in range(active_x_planes):
        active_slots = min(ci_repl, batch_len - plane * ci_repl)
        active_parts = active_slots * ci_stride
        xp = ws["x_padded"][plane][:active_parts, :padded_volume].reshape((active_parts, pd, ph, pw))
        if valid_volume == 0:
            nisa.memset(dst=xp, value=0.0)
            continue
        if rd0 > 0:
            nisa.memset(dst=xp[:, :rd0, :, :], value=0.0)
        if rd0 + vd < pd:
            nisa.memset(dst=xp[:, rd0 + vd :, :, :], value=0.0)
        if rh0 > 0:
            nisa.memset(dst=xp[:, rd0 : rd0 + vd, :rh0, :], value=0.0)
        if rh0 + vh < ph:
            nisa.memset(dst=xp[:, rd0 : rd0 + vd, rh0 + vh :, :], value=0.0)
        if rw0 > 0:
            nisa.memset(dst=xp[:, rd0 : rd0 + vd, rh0 : rh0 + vh, :rw0], value=0.0)
        if rw0 + vw < pw:
            nisa.memset(dst=xp[:, rd0 : rd0 + vd, rh0 : rh0 + vh, rw0 + vw :], value=0.0)
        if not fold:
            nisa.dma_copy(
                dst=xp[:, rd0 : rd0 + vd, rh0 : rh0 + vh, rw0 : rw0 + vw],
                src=ws["x"][plane][:active_parts, :valid_volume].reshape((active_parts, vd, vh, vw)),
            )

    if valid_volume > 0 and fold:
        # No dense x staging: the HBM read lands straight in the padded interior the memsets framed.
        for j in range(batch_len):
            plane, slot = j // ci_repl, j % ci_repl
            xpb = ws["x_padded"][plane][slot * ci_stride : slot * ci_stride + ci_len, :padded_volume].reshape(
                (ci_len, pd, ph, pw)
            )
            nisa.dma_copy(dst=xpb[:, rd0 : rd0 + vd, rh0 : rh0 + vh, rw0 : rw0 + vw], src=_x_src(j))


def _dw_build_xt(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    ws_idx: int,
    xt_slot: int,
    x_plane: int,
    tap: int,
    batch_len: int,
    pos0: int,
    pos_len: int,
) -> None:
    """Transpose one tap from x_padded into the x_t ring."""
    ws = tensors["working_sets"][ws_idx]
    packed_ci = tensors["packed_ci"]
    rows, d0, d1, h0, h1, w0, w1, _, _, _, _, _, _ = _dw_window_geometry(cfg, pos0, pos_len)
    pd, ph, pw = d1 - d0, h1 - h0, w1 - w0
    xp = ws["x_padded"][x_plane][:packed_ci, : pd * ph * pw].reshape((packed_ci, pd, ph, pw))
    xt = ws["xt"][xt_slot][x_plane]
    kd, kh, kw = _decode_spatial(tap, cfg.K_d, cfg.K_h, cfg.K_w)
    if tensors["row_reuse"]:
        # Build the k_h == 0 variant over an extended row range; the taps sharing this k_w read
        # it at a block offset instead (see _dw_row_reuse_params).
        kh = 0
        rows = _dw_row_reuse_rows(cfg, pos0, pos_len)
    for off, do, ho, wo, count in rows:
        id_ = do * cfg.stride_d + kd * cfg.dilation_d - cfg.pad_d_left
        ih = ho * cfg.stride_h + kh * cfg.dilation_h - cfg.pad_h_top
        iw = wo * cfg.stride_w + kw * cfg.dilation_w - cfg.pad_w_left
        kernel_assert(0 <= ih - h0 < ph, "dw x_t source row falls outside the loaded x window")
        src = (
            xp.select(dim=1, index=id_ - d0)
            .select(dim=1, index=ih - h0)
            .slice(dim=1, start=iw - w0, end=iw - w0 + (count - 1) * cfg.stride_w + 1, step=cfg.stride_w)
        )
        _dw_transpose_row(
            xt,
            src,
            off,
            count,
            packed_ci,
            partial_tmp=ws["x_tt"],
            row_tmp=ws["x_row"],
            contiguous=cfg.stride_w == 1,
        )


def _dw_band_kwargs(tensors: dict[str, object], j: int) -> dict[str, tuple[int, int]]:
    """PE column tile for batch band j, shared by the matmuls and by the primes that open them.

    A matmul's dst covers ci_len partitions from the band's offset j * ci_stride, and PE column c
    feeds PSUM partition c, so the column tile has to be the band. ci_stride is always one of
    32/64/128, the legal column-group widths. One band means the full array, which is the default.
    """
    ci_repl, ci_stride = tensors["ci_repl"], tensors["ci_stride"]
    if ci_repl <= 1:
        return {}
    return {
        "tile_size": (nl.tile_size.pmax, ci_stride),
        "tile_position": (0, (j % ci_repl) * ci_stride),
    }


def _dw_taps_per_pass(tensors: dict[str, object], co_len: int) -> int:
    """Taps whose accumulators fit PSUM at once: co_len fp32 columns each, spread over the banks."""
    return _PSUM_BANK_FMAX // co_len * _NUM_PSUM_BANKS // tensors["x_planes"]


def _dw_psum_grid(
    tensors: dict[str, object], co_len: int, tap_len: int, batch_slots: int
) -> list[list[list[nl.NkiTensor]]]:
    """Allocate the dW accumulator grid as one tensor per (batch band, PSUM bank), primed once each.

    accumulate=False clears the has_written status bits of the entire bank on the col_grp's
    partitions, so a group opening cannot be narrower than that. Band b's col_grp is exactly
    partitions [b*ci_stride, (b+1)*ci_stride), so a tiled prime over one bank of one band clears
    precisely that tensor: every (band, bank) pair is disjoint, each gets one accumulate=False, and
    every real matmul then lands on a set bit and truly accumulates. Nothing relies on a cleared bit
    silently converting an accumulate into an overwrite.

    Bands and taps only overlap when each is its own memloc rather than a slice of a shared one, so
    spread the slots across all the banks instead of densely filling each.
    """
    x_planes = tensors["x_planes"]
    ci_repl, ci_stride = tensors["ci_repl"], tensors["ci_stride"]
    slots = tap_len * x_planes
    # min() is load-bearing: it turns an over-long tap_len into the assert below rather than a
    # silent bank overflow.
    per_bank = min(_PSUM_BANK_FMAX // co_len, div_ceil(slots, _NUM_PSUM_BANKS))
    n_banks = div_ceil(slots, per_bank)
    n_bands = min(ci_repl, batch_slots)  # higher bands are never written, so never primed
    kernel_assert(n_banks <= _NUM_PSUM_BANKS, "dw accumulator grid exceeds psum")

    # regions[band][bank], ci_stride partitions tall at the band's offset, its live slots wide
    regions = []
    for band in range(n_bands):
        kwargs = _dw_band_kwargs(tensors, band)
        banks = []
        for bank in range(n_banks):
            width = min(per_bank, slots - bank * per_bank) * co_len
            banks.append(
                nl.ndarray(
                    shape=(ci_stride, width),
                    dtype=_ACC_DTYPE,
                    buffer=nl.psum,
                    address=(band * ci_stride, bank * _PSUM_BANK_FMAX * _ACC_DTYPE_SIZE),
                )
            )
            nisa.nc_matmul(
                dst=banks[bank],
                stationary=tensors["zero"][:, :ci_stride],
                moving=tensors["zero"][:, :width],
                accumulate=False,
                **kwargs,
            )
        regions.append(banks)

    # acc[tap][plane][band]: taps and planes pick the (bank, column), bands pick the region.
    acc = []
    for tap_idx in range(tap_len):
        planes = []
        for plane in range(x_planes):
            slot = tap_idx * x_planes + plane
            bank, col = slot // per_bank, (slot % per_bank) * co_len
            bands = []
            for band in range(n_bands):
                bands.append(regions[band][bank][:, col : col + co_len])
            planes.append(bands)
        acc.append(planes)
    return acc


def _dw_compute(
    tensors: dict[str, object],
    ws_idx: int,
    xt_slot: int,
    acc: list[list[nl.NkiTensor]],
    batch_len: int,
    pos_len: int,
    ci_len: int,
    co_len: int,
    kh_blocks: int = 0,
) -> None:
    """Contract one work item into the tap's PSUM slots."""
    ws = tensors["working_sets"][ws_idx]
    ci_repl, ci_stride = tensors["ci_repl"], tensors["ci_stride"]
    co_repl, co_stride = tensors["co_repl"], tensors["co_stride"]
    packed_ci, packed_co = tensors["packed_ci"], tensors["packed_co"]
    blocks = div_ceil(pos_len, nl.tile_size.pmax)
    kernel_assert(blocks + kh_blocks <= tensors["xt_blocks"], "dw x_t block offset overruns the buffer")
    # Per band, all loop-invariant in block: the dst slice, its PE column tile, and the operand
    # bands. Band innermost below, so consecutive matmuls rotate across the column tiles instead of
    # queueing `blocks` deep on one dst -- free to order either way because every matmul accumulates
    # onto a bank _dw_psum_grid already primed.
    bands = []
    for j in range(batch_len):
        x_plane, x_slot = j // ci_repl, j % ci_repl
        y_plane, y_slot = j // co_repl, j % co_repl
        bands.append(
            (
                acc[x_plane][x_slot][:ci_len, :co_len],
                x_plane,
                x_slot * ci_stride,
                y_plane,
                y_slot * co_stride,
                _dw_band_kwargs(tensors, j),
            )
        )
    for block in range(blocks):
        k_len = min(nl.tile_size.pmax, pos_len - block * nl.tile_size.pmax)
        for dst, x_plane, xt_band, y_plane, yt_band, kwargs in bands:
            xt_col0 = (block + kh_blocks) * packed_ci + xt_band
            yt_col0 = block * packed_co + yt_band
            stationary = ws["xt"][xt_slot][x_plane][:k_len, xt_col0 : xt_col0 + ci_len]
            moving = ws["yt"][y_plane][:k_len, yt_col0 : yt_col0 + co_len]
            nisa.nc_matmul(
                dst=dst,
                stationary=stationary,
                moving=moving,
                accumulate=True,
                **kwargs,
            )


def _dw_store(
    cfg: Conv3dBwdConfig,
    tensors: dict[str, object],
    dw: nl.NkiTensor,
    acc: list[list[nl.NkiTensor]],
    tap: int,
    ci0: int,
    ci_len: int,
    co0: int,
    co_len: int,
    active_batch_slots: int,
    batch_shard: bool = False,
) -> None:
    """Reduce batch and LNC partials, then store one dw tap tile."""
    ci_repl = tensors["ci_repl"]
    reduced = tensors["reduced"][:ci_len, :co_len]
    first = True
    for j in range(active_batch_slots):
        x_plane, x_slot = j // ci_repl, j % ci_repl
        src = acc[x_plane][x_slot][:ci_len, :co_len]
        if first:
            nisa.tensor_copy(dst=reduced, src=src)
            first = False
        else:
            nisa.tensor_tensor(dst=reduced, data1=reduced, data2=src, op=nl.add)
    if batch_shard:
        recv = tensors["recv"][:ci_len, :co_len]
        other = 1 - cfg.prg_id
        nisa.sendrecv(src=reduced, dst=recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.tensor_tensor(dst=reduced, data1=reduced, data2=recv, op=nl.add)
    result = tensors["result"][:ci_len, :co_len]
    nisa.tensor_copy(dst=result, src=reduced)
    kd, kh, kw = _decode_spatial(tap, cfg.K_d, cfg.K_h, cfg.K_w)
    dst = _dw_hbm_view(dw, kd, kh, kw, ci0, ci0 + ci_len, co0, co0 + co_len)
    nisa.dma_copy(dst=dst, src=result)


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
