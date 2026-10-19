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

"""3D Convolution backward kernel for NeuronCore.

Computes any subset of {dx, dw, db} for a 3D convolution. The kernel is organized as a set of
planner-selected components, a config's geometry chooses which path runs for each
gradient."""

import dataclasses

import nki
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger
from .conv3d_bwd_helpers import (
    _ACC_DTYPE_SIZE,
    _NUM_PSUM_BANKS,
    _SBUF_HEADROOM,
    BwdStrategy,
    Conv3dBwdConfig,
    Conv3dBwdCtx,
    DwFastVariant,
    DwUnitVariant,
    DwXLoadMethod,
    DxMethod,
    _dw_alloc_tensors,
    _dw_compute,
    _dw_free_tensors,
    _dw_layout,
    _dw_load_dy,
    _dw_load_x,
    _dw_store,
    _forward_out_dim,
)
from .conv3d_bwd_kernels import (
    _conv3d_db,
    _conv3d_dw_batched,
    _conv3d_dw_fast,
    _conv3d_dw_fast_packed,
    _conv3d_dw_phasesplit,
    _conv3d_dw_pointwise,
    _conv3d_dw_pointwise_strided,
    _conv3d_dx_col2im,
    _conv3d_dx_gemm,
    _conv3d_dx_gemm2d,
    _conv3d_dx_pointwise,
    _conv3d_dx_pointwise_strided,
    _dw_fast_prefetch,
    _dw_fast_prologue_load,
    _dw_fast_reserve,
    _dwps_alloc,
    _dwps_compute,
    _dwps_prefetch,
    _finalize_db_pointwise_strided,
    _issue_db_sendrecv,
    _pw_alloc,
    _pw_alloc_pk,
    _pw_db,
    _pw_dw,
    _pw_dw_pk,
    _pw_dx,
    _pw_dx_dw_fused,
    _pw_load_dy_resident,
    _pw_load_filters_t,
    _pw_store_dw_db,
    _sw_alloc,
    _sw_db_accum,
    _sw_db_finish,
    _sw_db_init,
    _sw_dw,
    _sw_dx,
    _sw_load_operands,
    _sw_store_dw,
)


@nki.jit
def conv3d_bwd(
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    compute_dx: bool = True,
    compute_dw: bool = True,
    compute_db: bool = True,
) -> tuple[nl.NkiTensor, ...]:
    """
    3D Convolution backward pass.

    Computes any subset of {dx, dw, db} for a 3D convolution forward pass. The planner
    selects a strategy and per-gradient compute method from the problem geometry.

    Dimensions:
        B: Batch size
        C_in: Number of input channels
        C_out: Number of output channels
        D, H, W: Input spatial dimensions
        K_d, K_h, K_w: Filter spatial dimensions
        D_out, H_out, W_out: Output spatial dimensions

    Args:
        dy (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], Output gradient on HBM.
        x_in (nl.NkiTensor): [B, C_in, D, H, W], Forward input on HBM.
        filters (nl.NkiTensor): [K_d, K_h, K_w, C_in, C_out], Forward filters on HBM.
        stride (tuple[int, int, int]): (stride_d, stride_h, stride_w), Forward convolution strides.
        padding (tuple[int, int, int, int, int, int]): (pad_d_left, pad_d_right, pad_h_top,
            pad_h_bottom, pad_w_left, pad_w_right), Forward padding.
        dilation (tuple[int, int, int]): (dilation_d, dilation_h, dilation_w), Forward dilation factors.
        compute_dx (bool): Whether to compute the input gradient dx.
        compute_dw (bool): Whether to compute the filter gradient dw.
        compute_db (bool): Whether to compute the bias gradient db.

    Returns:
        tuple[nl.NkiTensor, ...]: The requested gradient tensors in canonical order
        (dx, dw, db), including only those selected by the compute_* flags:
            - dx: [B, C_in, D, H, W]
            - dw: [K_d, K_h, K_w, C_in, C_out]
            - db: [C_out]

    Notes:
        - Supports the same 6-value (possibly asymmetric) padding as the forward conv3d
          kernel for all of dx, dw, and db.
        - The planner (see the PLANNER section and the module docstring's dispatch map)
          selects the strategy and per-gradient compute method from the problem geometry.

    Pseudocode:
        cfg = build_conv3d_bwd_config(dy, x_in, filters, stride, padding, dilation, ...)
        if cfg.strategy == PER_GRADIENT:
            # Each requested gradient loads its own dy and runs its planned component.
            out = _run_per_gradient(cfg, dy, x_in, filters)
        else:
            # dy stays resident and is shared across dx/dw/db.
            ctx = Conv3dBwdCtx(cfg, dy, x_in, filters)
            if cfg.strategy == SHARE_DY_STRIDED:
                _run_shared_dy_strided(ctx)      # stride>1: scatter dx onto the stride grid
            elif cfg.dw_unit_variant == PACKED:
                _run_shared_dy_unit_pk(ctx)      # stride 1: packed two-phase accumulate
            else:
                _run_shared_dy_unit(ctx)         # stride 1: fused or scheduled non-fused
            out = (ctx.dx, ctx.dw, ctx.db) selected by compute_* flags
        return the single tensor if only one gradient requested, else the tuple
    """

    cfg = build_conv3d_bwd_config(
        dy=dy,
        x_in=x_in,
        filters=filters,
        stride=stride,
        padding=padding,
        dilation=dilation,
        compute_dx=compute_dx,
        compute_dw=compute_dw,
        compute_db=compute_db,
    )

    # Dispatch on the planner-chosen dy-residency strategy
    if cfg.strategy == BwdStrategy.PER_GRADIENT:
        out = _run_per_gradient(cfg, dy, x_in, filters)
    else:
        ctx = Conv3dBwdCtx(cfg, dy, x_in, filters)
        if cfg.strategy == BwdStrategy.SHARE_DY_STRIDED:
            _run_shared_dy_strided(ctx)
        elif cfg.dw_unit_variant == DwUnitVariant.PACKED:
            _run_shared_dy_unit_pk(ctx)
        else:
            _run_shared_dy_unit(ctx)
        out = (ctx.dx, ctx.dw, ctx.db) if cfg.compute_db else (ctx.dx, ctx.dw)

    return out[0] if len(out) == 1 else out


# NKI 33523c02 opts kernels out of non-SSA legalization by default. Conv3DBwd's
# dw path still relies on that backend transformation for correct numerics.
if hasattr(conv3d_bwd, "enable_non_ssa_legalization"):
    conv3d_bwd = conv3d_bwd.enable_non_ssa_legalization()


def _pointwise_supported(cfg: "Conv3dBwdConfig") -> bool:
    """True when the 1x1 pointwise dw fast path applies."""
    return cfg.n_taps == 1 and cfg.padding == (0, 0, 0, 0, 0, 0) and cfg.dilation == (1, 1, 1)


def _batched_gemm_supported(cfg: "Conv3dBwdConfig") -> bool:
    """True when the batched-GEMM dw path applies."""
    if cfg.dilation != (1, 1, 1):
        return False
    P_MAX = nl.tile_size.pmax
    ds = cfg.dtype_size
    spatial_in = cfg.D * cfg.H * cfg.W
    spatial_out = cfg.D_out * cfg.H_out * cfg.W_out
    ci_tile = min(P_MAX, cfg.C_in)
    co_tile = min(P_MAX, cfg.C_out)
    n_ci = div_ceil(cfg.C_in, ci_tile)
    n_co = div_ceil(cfg.C_out, co_tile)
    pack = 1 if spatial_out >= P_MAX else min(cfg.B, max(1, div_ceil(4 * P_MAX, spatial_out)))
    L = pack * spatial_out
    n_kt = div_ceil(L, P_MAX)
    bytes_pp = (
        n_co * L * ds  # dyc
        + n_ci * pack * spatial_in * ds  # x_full
        + n_ci * L * ds  # xc
        + n_co * n_kt * co_tile * ds  # dy_pos
        + n_ci * n_kt * ci_tile * ds  # x_pos
        + P_MAX * ds  # identity
        + ci_tile * ds  # result
        + cfg.n_taps * n_co * n_ci * ci_tile * _ACC_DTYPE_SIZE  # dw_acc (fp32, all taps resident)
    )
    return bytes_pp <= nl.tile_size.total_available_sbuf_size - _SBUF_HEADROOM


def _phase_split_supported(cfg: "Conv3dBwdConfig") -> bool:
    """True when the stride-2 phase-split x-load is applicable (see DwXLoadMethod)."""
    P_MAX = nl.tile_size.pmax
    if cfg.D != 1 or cfg.K_d != 1 or cfg.stride_d != 1:
        return False
    if cfg.stride_h != 2 or cfg.stride_w != 2:
        return False
    if cfg.dilation != (1, 1, 1):
        return False
    if cfg.C_out > P_MAX:
        return False
    if cfg.C_in > P_MAX // 2:
        return False
    return True


def _plan_dw_x_load(cfg: "Conv3dBwdConfig") -> DwXLoadMethod:
    """Select the dw x-load method from the problem geometry."""
    if _pointwise_supported(cfg):
        return DwXLoadMethod.POINTWISE
    if _phase_split_supported(cfg):
        return DwXLoadMethod.PHASE_SPLIT
    if _dw_fast_supported(cfg):
        return DwXLoadMethod.FAST_PLANE
    if _batched_gemm_supported(cfg):
        return DwXLoadMethod.BATCHED_GEMM
    return DwXLoadMethod.GATHER_TRANSPOSE


def _plan_dx(cfg: "Conv3dBwdConfig") -> "DxMethod":
    """Resolve the dx compute component fully from geometry."""
    if _dx_gemm_supported(cfg):
        return DxMethod.GEMM
    if cfg.n_taps == 1 and cfg.padding == (0, 0, 0, 0, 0, 0) and cfg.stride == (1, 1, 1) and cfg.dilation == (1, 1, 1):
        return DxMethod.POINTWISE
    if _dx_pointwise_strided_supported(cfg):
        return DxMethod.POINTWISE_STRIDED
    if _dx_gemm2d_supported(cfg):
        return DxMethod.GEMM2D
    return DxMethod.COL2IM


def _plan_dw_fast_variant(cfg: "Conv3dBwdConfig") -> "DwFastVariant":
    """Resolve which FAST_PLANE dw runs."""
    P_MAX = nl.tile_size.pmax
    ci_tile = min(P_MAX, cfg.C_in)
    if ci_tile * 2 <= P_MAX and ci_tile % 32 == 0 and cfg.n_taps >= 2:
        return DwFastVariant.PACKED
    return DwFastVariant.GENERAL


def _plan_dw_unit_variant(cfg: "Conv3dBwdConfig") -> "DwUnitVariant":
    """Resolve which SHARE_DY_POINTWISE unit-path dw runs."""
    n_cb = div_ceil(cfg.C_out, min(nl.tile_size.psum_fmax, cfg.C_out))
    n_ci = div_ceil(cfg.C_in, min(nl.tile_size.pmax, cfg.C_in))
    n_co = div_ceil(cfg.C_out, min(nl.tile_size.pmax, cfg.C_out))
    ratio_ok = (cfg.D * cfg.H * cfg.W) >= 16 * n_ci * n_cb
    # PACKED: large-C/small-plane that only fits via residency (fails ratio); uses two-phase accumulate.
    bank_oversub = n_ci * n_cb > _NUM_PSUM_BANKS
    small_plane = cfg.spatial_out < nl.tile_size.pmax
    if (not ratio_ok) and (n_cb >= 2 or bank_oversub or small_plane) and _pw_residency_fits(cfg, n_ci, n_cb):
        return DwUnitVariant.PACKED
    # FUSED: large resident plane with multiple co-tiles -> double-buffer dy across batches.
    if cfg.spatial_out >= 2048 and n_co > 1 and cfg.dtype_size == 2:
        return DwUnitVariant.FUSED
    return DwUnitVariant.BASELINE


def _pw_residency_fits(cfg: "Conv3dBwdConfig", n_ci: int, n_cb: int) -> bool:
    """True when the SHARE_DY_POINTWISE resident set (dy + dw fp32 accumulators + filtersT) fits SBUF, so dy can stay resident and be shared across dx/dw/db instead of streaming it twice."""
    P_MAX = nl.tile_size.pmax
    ds = cfg.dtype_size
    plane = cfg.D * cfg.H * cfg.W
    Bp = div_ceil(cfg.B, cfg.n_prgs)
    ci_tile = min(P_MAX, cfg.C_in)
    co_tile = min(P_MAX, cfg.C_out)
    co_blk = min(nl.tile_size.psum_fmax, cfg.C_out)
    n_co = div_ceil(cfg.C_out, co_tile)
    dy_res = n_co * Bp * plane * ds
    dw_sbuf = n_ci * n_cb * co_blk * _ACC_DTYPE_SIZE
    filters_t = n_ci * n_co * ci_tile * ds
    identity = P_MAX * ds
    n_pt = div_ceil(plane, P_MAX)
    xt_res = Bp * n_pt * cfg.C_in * ds
    dyt_res = Bp * n_pt * cfg.C_out * ds
    total = dy_res + dw_sbuf + filters_t + identity + xt_res + dyt_res
    return total <= nl.tile_size.total_available_sbuf_size - _SBUF_HEADROOM


def _plan_strategy(cfg: "Conv3dBwdConfig") -> "BwdStrategy":
    """Pick the top-level backward strategy from the geometry (see BwdStrategy)."""
    if not (
        cfg.compute_dx
        and cfg.compute_dw
        and cfg.n_taps == 1
        and cfg.padding == (0, 0, 0, 0, 0, 0)
        and cfg.dilation == (1, 1, 1)
        and cfg.dtype_size == 2
        and cfg.n_prgs in (1, 2)
    ):
        return BwdStrategy.PER_GRADIENT
    n_cb = div_ceil(cfg.C_out, min(nl.tile_size.psum_fmax, cfg.C_out))
    n_ci = div_ceil(cfg.C_in, min(nl.tile_size.pmax, cfg.C_in))
    ratio_ok = (cfg.D * cfg.H * cfg.W) >= 16 * n_ci * n_cb
    bank_oversub = n_ci * n_cb > _NUM_PSUM_BANKS
    small_plane = cfg.spatial_out < nl.tile_size.pmax
    packed_ok = (n_cb >= 2 or bank_oversub or small_plane) and _pw_residency_fits(cfg, n_ci, n_cb)
    if cfg.stride == (1, 1, 1) and (ratio_ok or packed_ok):
        return BwdStrategy.SHARE_DY_POINTWISE
    banks_min = min(n_ci * n_cb, div_ceil(n_ci, 2) * n_cb, n_ci * div_ceil(n_cb, 2))
    if banks_min <= _NUM_PSUM_BANKS - 2:
        return BwdStrategy.SHARE_DY_STRIDED
    return BwdStrategy.PER_GRADIENT


_INTERLEAVE_BY_STRATEGY = {
    BwdStrategy.SHARE_DY_POINTWISE: {},
    BwdStrategy.SHARE_DY_STRIDED: {"operands": 2, "sw_out": 2},
    BwdStrategy.PER_GRADIENT: {"dy": 2, "x": 2, "pipe": 3, "dy_t": 2},
}
_INTERLEAVE_BY_DW_METHOD = {}
_INTERLEAVE_BY_DX_METHOD = {}


def _plan_interleave(cfg: "Conv3dBwdConfig") -> dict:
    """Buffering degrees for the chosen strategy plus the per-gradient components it runs."""
    degrees = dict(_INTERLEAVE_BY_STRATEGY[cfg.strategy])
    degrees.update(_INTERLEAVE_BY_DW_METHOD.get(cfg.dw_x_load_method, {}))
    degrees.update(_INTERLEAVE_BY_DX_METHOD.get(cfg.dx_method, {}))
    return degrees


def build_conv3d_bwd_config(
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
    dilation: tuple[int, int, int],
    compute_dx: bool,
    compute_dw: bool,
    compute_db: bool,
) -> Conv3dBwdConfig:
    """Validate the raw kernel inputs and unpack them into a :class:`Conv3dBwdConfig`."""
    kernel_assert(len(x_in.shape) == 5, f"x_in must be 5D [B, C_in, D, H, W], got {x_in.shape}")
    kernel_assert(len(dy.shape) == 5, f"dy must be 5D [B, C_out, D_out, H_out, W_out], got {dy.shape}")
    kernel_assert(len(filters.shape) == 5, f"filters must be 5D [K_d, K_h, K_w, C_in, C_out], got {filters.shape}")
    kernel_assert(len(stride) == 3, f"stride must be a 3-tuple, got {stride}")
    kernel_assert(len(padding) == 6, f"padding must be a 6-tuple, got {padding}")
    kernel_assert(len(dilation) == 3, f"dilation must be a 3-tuple, got {dilation}")
    kernel_assert(
        compute_dx or compute_dw or compute_db,
        "At least one of compute_dx, compute_dw, compute_db must be True.",
    )

    B, C_in, D, H, W = x_in.shape
    K_d, K_h, K_w, filt_c_in, C_out = filters.shape
    dy_B, dy_C_out, D_out, H_out, W_out = dy.shape

    kernel_assert(filt_c_in == C_in, f"C_in mismatch: x_in={C_in} vs filters={filt_c_in}")
    kernel_assert(dy_B == B, f"Batch mismatch: x_in={B} vs dy={dy_B}")
    kernel_assert(dy_C_out == C_out, f"C_out mismatch: filters={C_out} vs dy={dy_C_out}")
    kernel_assert(x_in.dtype == filters.dtype, f"dtype mismatch: x_in={x_in.dtype} vs filters={filters.dtype}")
    kernel_assert(x_in.dtype == dy.dtype, f"dtype mismatch: x_in={x_in.dtype} vs dy={dy.dtype}")

    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation

    exp_D_out = _forward_out_dim(D, K_d, stride_d, pad_d_left, pad_d_right, dilation_d)
    exp_H_out = _forward_out_dim(H, K_h, stride_h, pad_h_top, pad_h_bottom, dilation_h)
    exp_W_out = _forward_out_dim(W, K_w, stride_w, pad_w_left, pad_w_right, dilation_w)
    kernel_assert(
        D_out == exp_D_out and H_out == exp_H_out and W_out == exp_W_out,
        f"dy spatial mismatch: dy=({D_out},{H_out},{W_out}) vs expected ({exp_D_out},{exp_H_out},{exp_W_out})",
    )

    grid_ndim, n_prgs, prg_id = get_verified_program_sharding_info("conv3d_bwd_fused", (0, 1))

    cfg = Conv3dBwdConfig(
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
        compute_dx=compute_dx,
        compute_dw=compute_dw,
        compute_db=compute_db,
        grid_ndim=grid_ndim,
        n_prgs=n_prgs,
        prg_id=prg_id,
        interleave={},
        dw_x_load_method=DwXLoadMethod.GATHER_TRANSPOSE,
        dx_method=DxMethod.COL2IM,
        dtype_size=sizeinbytes(x_in.dtype),
        strategy=BwdStrategy.PER_GRADIENT,
        dw_fast_variant=DwFastVariant.GENERAL,
        dw_unit_variant=DwUnitVariant.BASELINE,
    )
    cfg = dataclasses.replace(
        cfg,
        dw_x_load_method=_plan_dw_x_load(cfg),
        dx_method=_plan_dx(cfg),
        strategy=_plan_strategy(cfg),
    )
    cfg = dataclasses.replace(
        cfg,
        dw_fast_variant=_plan_dw_fast_variant(cfg),
        dw_unit_variant=_plan_dw_unit_variant(cfg),
    )
    cfg = dataclasses.replace(cfg, interleave=_plan_interleave(cfg))

    logger = get_logger("conv3d_bwd")
    logger.info(
        f"conv3d_bwd: B={cfg.B}, C_in={cfg.C_in}, C_out={cfg.C_out}, "
        f"in=({cfg.D},{cfg.H},{cfg.W}), K=({cfg.K_d},{cfg.K_h},{cfg.K_w}), "
        f"out=({cfg.D_out},{cfg.H_out},{cfg.W_out}), stride={cfg.stride}, "
        f"padding={cfg.padding}, dilation={cfg.dilation}, "
        f"dx={cfg.compute_dx}, dw={cfg.compute_dw}, db={cfg.compute_db}, "
        f"n_prgs={cfg.n_prgs}, prg_id={cfg.prg_id}"
    )

    return cfg


def _run_shared_dy_unit_pk(ctx):
    """Shared-dy, stride 1: dy resident once, shared by dx/dw/db."""
    _pw_alloc_pk(ctx)
    _pw_load_dy_resident(ctx)
    _pw_load_filters_t(ctx)
    _pw_dx(ctx)
    _pw_dw_pk(ctx)  # db reduction is interleaved inside its phase-2 loop
    _pw_store_dw_db(ctx)


def _dw_fast_supported(cfg: Conv3dBwdConfig) -> bool:
    """True when the resident-plane fast dw path applies (see _conv3d_dw_fast)."""
    if cfg.dilation != (1, 1, 1):
        return False
    if not (cfg.n_taps > 1 or (cfg.C_in <= nl.tile_size.pmax and cfg.C_out <= nl.tile_size.pmax)):
        return False
    P_MAX = nl.tile_size.pmax
    ds = cfg.dtype_size
    spatial_in = cfg.D * cfg.H * cfg.W
    spatial_out = cfg.D_out * cfg.H_out * cfg.W_out
    ci_tile = min(P_MAX, cfg.C_in)
    co_tile = min(P_MAX, cfg.C_out)
    n_co = div_ceil(cfg.C_out, co_tile)
    n_pos = div_ceil(spatial_out, P_MAX)
    fast_bytes = (
        spatial_in * ds  # x_plane
        + n_co * spatial_out * ds  # y_plane
        + n_co * n_pos * co_tile * ds  # yt
        + spatial_out * ds  # xs
        + n_pos * ci_tile * ds  # xt
        + cfg.n_taps * n_co * co_tile * _ACC_DTYPE_SIZE  # dw_acc (fp32)
        + P_MAX * P_MAX * ds  # identity
    )
    return fast_bytes <= nl.tile_size.total_available_sbuf_size - _SBUF_HEADROOM


def _conv3d_dw(
    cfg: Conv3dBwdConfig,
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    sbm: SbufManager,
    prologue=None,
    want_db=False,
    prefetch=None,
):
    """Compute dw = [K_d, K_h, K_w, C_in, C_out] via the tiled dy-stationary GEMM."""
    if cfg.dw_x_load_method == DwXLoadMethod.POINTWISE:
        spatial_out = cfg.D_out * cfg.H_out * cfg.W_out
        packable = spatial_out < nl.tile_size.pmax and cfg.B > 1
        if cfg.stride != (1, 1, 1) or packable:
            return _conv3d_dw_pointwise_strided(cfg, dy, x_in, sbm), None
        return _conv3d_dw_pointwise(cfg, dy, x_in, sbm), None
    if cfg.dw_x_load_method == DwXLoadMethod.PHASE_SPLIT:
        return _conv3d_dw_phasesplit(cfg, dy, x_in, sbm), None
    if cfg.dw_x_load_method == DwXLoadMethod.FAST_PLANE:
        if cfg.dw_fast_variant == DwFastVariant.PACKED:
            return _conv3d_dw_fast_packed(cfg, dy, x_in, sbm, prologue=prologue), None
        return _conv3d_dw_fast(cfg, dy, x_in, sbm, want_db=want_db, prefetch=prefetch)
    if cfg.dw_x_load_method == DwXLoadMethod.BATCHED_GEMM:
        return _conv3d_dw_batched(cfg, dy, x_in, sbm), None

    dtype = x_in.dtype
    dw_dtype = dtype

    dw = nl.ndarray(shape=(cfg.K_d, cfg.K_h, cfg.K_w, cfg.C_in, cfg.C_out), dtype=dw_dtype, buffer=nl.shared_hbm)

    tensors = _dw_alloc_tensors(cfg, sbm, dtype, dw_dtype)

    pos_tile = cfg.pos_tile
    cout_tile = cfg.cout_tile
    n_cols_tile = cfg.n_cols_tile
    spatial_out = cfg.D_out * cfg.H_out * cfg.W_out

    all_tiles = [(co0, col0) for co0 in range(0, cfg.C_out, cout_tile) for col0 in range(0, cfg.n_cols, n_cols_tile)]
    my_tiles = all_tiles[cfg.prg_id :: cfg.n_prgs] if cfg.is_sharded else all_tiles

    bank = 0
    for co0, col0 in my_tiles:
        co_len = min(cout_tile, cfg.C_out - co0)
        col_len = min(n_cols_tile, cfg.n_cols - col0)
        bank_idx = bank % _NUM_PSUM_BANKS
        bank += 1

        first = True
        pp = 0
        for batch_idx in range(cfg.B):
            for pos0 in range(0, spatial_out, pos_tile):
                pos_len = min(pos_tile, spatial_out - pos0)
                _dw_load_dy(cfg, tensors, dy, pp, batch_idx, pos0, pos_len, co0, co_len)
                _dw_load_x(cfg, tensors, x_in, pp, batch_idx, pos0, pos_len, col0, col_len)
                _dw_compute(cfg, tensors, pp, pp, bank_idx, co_len, col_len, pos_len, accumulate=not first)
                first = False
                pp ^= 1

        _dw_layout(cfg, tensors, bank_idx, co_len, col_len)
        _dw_store(cfg, tensors, dw, co0, co_len, col0, col_len)

    _dw_free_tensors(sbm, tensors)
    return dw, None


def _dx_gemm_supported(cfg: Conv3dBwdConfig) -> bool:
    """True for the 2D stride-2 small-C_in stem where the tap-expand GEMM dx wins."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    if cfg.D != 1 or cfg.K_d != 1 or cfg.stride_d != 1:
        return False
    if cfg.stride_h != 2 or cfg.stride_w != 2:
        return False
    if cfg.dilation != (1, 1, 1):
        return False
    if cfg.C_in > P_MAX // 2 or cfg.C_out > P_MAX:
        return False
    if cfg.W_out > P_MAX:
        return False
    if cfg.B * cfg.C_in > F_MAX:
        return False
    return True


def _dx_gemm2d_residency_fits(cfg: "Conv3dBwdConfig", H_pad: int, W_pad: int, c_dy0: int) -> bool:
    """True when GEMM2D's persistent SBUF allocations fit with headroom."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    ds = cfg.dtype_size
    co_tile = min(P_MAX, cfg.C_out)
    n_co = div_ceil(cfg.C_out, co_tile)
    batches_per_program = div_ceil(cfg.B, cfg.n_prgs) if cfg.is_sharded else cfg.B
    nbuf = 2 if batches_per_program > 1 else 1

    pack_ok = n_co == 1 and co_tile <= 64 and cfg.K_w >= 2 and c_dy0 - cfg.dilation_w >= 0
    n_wt_pk = cfg.K_h * (cfg.K_w // 2) if pack_ok else 0
    band = max(1, F_MAX // cfg.W)

    total = (
        cfg.n_taps * n_co * cfg.C_in * ds  # wt[tap][co]
        + n_wt_pk * cfg.C_in * ds  # packed tap-pair weights
        + nbuf * n_co * H_pad * W_pad * ds  # zero-padded dy
        + nbuf * n_co * cfg.H_out * cfg.W_out * ds  # contiguous dy
        + band * cfg.W * ds  # output staging
    )
    if cfg.compute_dx and cfg.compute_db:
        total += 2 * n_co * _ACC_DTYPE_SIZE  # db_acc + db_tmp
        total += n_co * ds  # db_out
        if cfg.is_sharded:
            total += n_co * _ACC_DTYPE_SIZE  # db_recv
    return total <= nl.tile_size.total_available_sbuf_size - _SBUF_HEADROOM


def _dx_gemm2d_supported(cfg: "Conv3dBwdConfig") -> bool:
    """True when stride-1 2D GEMM is geometrically valid and fits in SBUF."""
    F_MAX = nl.tile_size.psum_fmax
    if cfg.D != 1 or cfg.K_d != 1 or cfg.stride_d != 1:
        return False
    if cfg.stride_h != 1 or cfg.stride_w != 1:
        return False
    if cfg.W > F_MAX:
        return False
    r_dy0 = (cfg.K_h - 1) * cfg.dilation_h - cfg.pad_h_top
    c_dy0 = (cfg.K_w - 1) * cfg.dilation_w - cfg.pad_w_left
    H_pad = cfg.H + (cfg.K_h - 1) * cfg.dilation_h
    W_pad = cfg.W + (cfg.K_w - 1) * cfg.dilation_w
    # The stored valid-dy interior must land inside the zero-padded buffer.
    if r_dy0 < 0 or c_dy0 < 0:
        return False
    if r_dy0 + cfg.H_out > H_pad or c_dy0 + cfg.W_out > W_pad:
        return False
    return _dx_gemm2d_residency_fits(cfg, H_pad, W_pad, c_dy0)


def _dx_pointwise_strided_supported(cfg: "Conv3dBwdConfig") -> bool:
    """True when dx is a 1x1x1 conv with stride>1 and no padding/dilation."""
    return (
        cfg.n_taps == 1 and cfg.padding == (0, 0, 0, 0, 0, 0) and cfg.dilation == (1, 1, 1) and cfg.stride != (1, 1, 1)
    )


def _db_foldable_into_dx_pointwise_strided(cfg: "Conv3dBwdConfig") -> bool:
    """True when db can be folded for free into the batch-packed pointwise-strided dx pass."""
    return cfg.compute_db and cfg.compute_dx and _dx_pointwise_strided_supported(cfg)


def _conv3d_dx(
    cfg: Conv3dBwdConfig, dy: nl.NkiTensor, filters: nl.NkiTensor, sbm: SbufManager, db: nl.NkiTensor = None
) -> nl.NkiTensor:
    """dx dispatcher: run the planner-chosen dx component (cfg.dx_method, see DxMethod)."""
    if cfg.dx_method == DxMethod.GEMM:
        return _conv3d_dx_gemm(cfg, dy, filters, sbm)
    if cfg.dx_method == DxMethod.POINTWISE:
        return _conv3d_dx_pointwise(cfg, dy, filters, sbm)
    if cfg.dx_method == DxMethod.POINTWISE_STRIDED:
        return _conv3d_dx_pointwise_strided(cfg, dy, filters, sbm)
    if cfg.dx_method == DxMethod.GEMM2D:
        return _conv3d_dx_gemm2d(cfg, dy, filters, sbm, db)
    return _conv3d_dx_col2im(cfg, dy, filters, sbm)


def _dx_fuses_db(cfg: "Conv3dBwdConfig") -> bool:
    """True when _conv3d_dx computes db as a fused side effect (the GEMM2D path)."""
    return cfg.dx_method == DxMethod.GEMM2D


def _run_shared_dy_strided(ctx):
    """Shared-dy, stride>1: dy resident per batch; dx scatters onto the stride grid; dw + db share it."""
    _sw_alloc(ctx)
    if ctx.compute_db:
        _sw_db_init(ctx)
    if ctx.Bp > 0:
        _sw_load_operands(ctx, ctx.b_lo, 0)
    cur, buf_sel = 0, 0
    for batch_local_idx in range(ctx.Bp):
        buf_sel = _sw_dx(ctx, batch_local_idx, cur, buf_sel)
        if batch_local_idx + 1 < ctx.Bp:
            _sw_load_operands(ctx, ctx.b_lo + batch_local_idx + 1, 1 - cur)  # prefetch next batch
        _sw_dw(ctx, batch_local_idx, cur)
        if ctx.compute_db:
            _sw_db_accum(ctx, cur)
        cur = 1 - cur
    _sw_store_dw(ctx)
    if ctx.compute_db:
        _sw_db_finish(ctx)


def _run_shared_dy_unit(ctx):
    """Shared-dy, stride 1: dy resident once, shared by dx/dw/db (fused or scheduled non-fused path)."""
    _pw_alloc(ctx)
    _pw_load_filters_t(ctx)
    if ctx.dy_double_buf:
        _pw_dx_dw_fused(ctx)
        if ctx.compute_db:
            _pw_db(ctx)
        _pw_store_dw_db(ctx)
        return
    _pw_load_dy_resident(ctx)
    if ctx.compute_db and ctx.pw_db_first:
        _pw_db(ctx)
    if ctx.pw_dx_last:
        _pw_dw(ctx)
        if ctx.compute_db and not ctx.pw_db_first:
            _pw_db(ctx)
        _pw_store_dw_db(ctx)
        _pw_dx(ctx)
        return
    _pw_dx(ctx)
    _pw_dw(ctx)
    if ctx.compute_db and not ctx.pw_db_first:
        _pw_db(ctx)
    _pw_store_dw_db(ctx)


def _run_per_gradient(cfg, dy, x_in, filters):
    """No dy sharing: each requested gradient loads its own dy and runs its planned component."""
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, logger=get_logger("conv3d_bwd"))
    out = []
    if cfg.dw_x_load_method == DwXLoadMethod.PHASE_SPLIT and cfg.compute_dw:
        fuse_db = cfg.compute_db
        if cfg.compute_dx:
            dw_state = _dwps_alloc(cfg, dy, x_in, sbm, compute_db=fuse_db)
            n_pf = _dwps_prefetch(cfg, dw_state)
            dx_out = _conv3d_dx(cfg, dy, filters, sbm)
            dw_res = _dwps_compute(cfg, dy, x_in, sbm, dw_state, compute_db=fuse_db, preloaded=n_pf)
            dw_t, db_fused = dw_res if fuse_db else (dw_res, None)
            out = [dx_out, dw_t]
        else:
            dw_res = _conv3d_dw_phasesplit(cfg, dy, x_in, sbm, compute_db=fuse_db)
            dw_t, db_fused = dw_res if fuse_db else (dw_res, None)
            out = [dw_t]
        if cfg.compute_db:
            out.append(db_fused if fuse_db else _conv3d_db(cfg, dy, sbm))
        return tuple(out)
    if cfg.dx_method == DxMethod.POINTWISE_STRIDED and cfg.compute_dx:
        fold_db = cfg.compute_db and _db_foldable_into_dx_pointwise_strided(cfg)
        db_defer = None
        if fold_db:
            dx, db_folded, db_part = _conv3d_dx_pointwise_strided(cfg, dy, filters, sbm, fold_db=True)
            db_defer = (db_folded, db_part)
        else:
            dx = _conv3d_dx_pointwise_strided(cfg, dy, filters, sbm)
        out.append(dx)
        if cfg.compute_dw:
            db_recv = _issue_db_sendrecv(cfg, db_defer[1], sbm) if db_defer != None else None
            out.append(_conv3d_dw(cfg, dy, x_in, sbm)[0])
            if db_defer != None:
                _finalize_db_pointwise_strided(cfg, dy, db_defer[0], db_defer[1], db_recv, sbm)
        elif db_defer != None:
            db_recv = _issue_db_sendrecv(cfg, db_defer[1], sbm)
            _finalize_db_pointwise_strided(cfg, dy, db_defer[0], db_defer[1], db_recv, sbm)
        if cfg.compute_db:
            out.append(db_defer[0] if fold_db else _conv3d_db(cfg, dy, sbm))
        return tuple(out)
    is_fast = cfg.dw_x_load_method == DwXLoadMethod.FAST_PLANE
    is_packed = is_fast and cfg.dw_fast_variant == DwFastVariant.PACKED
    db_via_dx = cfg.compute_db and cfg.compute_dx and _dx_fuses_db(cfg)
    db_via_dw = cfg.compute_db and cfg.compute_dw and is_fast and not is_packed and not db_via_dx
    db_hbm = nl.ndarray(shape=(cfg.C_out,), dtype=dy.dtype, buffer=nl.shared_hbm) if db_via_dx else None
    dw_prologue = dw_prefetch = None
    if cfg.compute_dx and cfg.compute_dw and is_fast:
        if is_packed:
            dw_prologue = _dw_fast_reserve(cfg, dy, x_in, sbm)
        elif cfg.dx_method != DxMethod.GEMM2D:
            dw_prefetch = _dw_fast_prefetch(cfg, dy, x_in, sbm)
    if cfg.compute_dx:
        out.append(_conv3d_dx(cfg, dy, filters, sbm, db_hbm))
    db_folded = None
    if cfg.compute_dw:
        if dw_prologue != None:
            _dw_fast_prologue_load(cfg, dy, x_in, dw_prologue)
            dw, _ = _conv3d_dw(cfg, dy, x_in, sbm, prologue=dw_prologue)
        else:
            dw, db_folded = _conv3d_dw(cfg, dy, x_in, sbm, want_db=db_via_dw, prefetch=dw_prefetch)
        out.append(dw)
    if dw_prologue != None:
        for _ in range(dw_prologue["n_pop"]):
            sbm.pop_heap()  # yt[..], x_plane0
    if cfg.compute_db:
        if db_via_dx:
            out.append(db_hbm)
        elif db_folded != None:
            out.append(db_folded)
        else:
            out.append(_conv3d_db(cfg, dy, sbm))
    return tuple(out)
