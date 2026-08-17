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

"""Generic pad kernel for tensors with constant, replicate, reflect, and circular modes.

Architecture overview::

    pad (top-level @nki.jit kernel)
     ├── _normalize_to_4d       → reshape to (NC, D, H, W) NkiTensors + PadParams
     ├── compute_tiling_strategy → tile sizes, sharding, SBUF padding policy
     └── 4 nested loops (NC, D, H, W):
          ├── tile_params        → per-tile PadParams (progressive narrowing)
          ├── _tile_input / _tile_output → NkiTensor slicing
          └── inner body:
               ├── DMA load     → contiguous input tile into SBUF
               ├── pad_compute  → interior copy + padding fill (W → H → D)
               └── DMA store    → contiguous padded tile to HBM
          After each loop level, deferred DMA padding fills dimensions
          that were not padded in SBUF (reflect/circular on multi-tiled dims).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from .pad_compute import pad_compute
from .pad_modes import PadMode, make_pad_mode
from .pad_params import PadParams
from .pad_sharding import shard_operation
from .pad_tiling import PadTilingStrategy, compute_tiling_strategy

# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _normalize_to_4d(x_ref, out_ref, padding: tuple, mode: str) -> tuple:
    """Reshape input/output to 4D ``(NC, D, H, W)`` NkiTensors and build PadParams.

    Supports arbitrary batch dimensions. The last ``len(padding) // 2``
    dimensions are treated as spatial; all preceding dimensions are collapsed
    into a single NC batch axis.

    Returns:
        ``(x_4d, out_4d, params)`` — NkiTensor pair and PadParams.
    """
    ndim = len(x_ref.shape)
    n_spatial = len(padding) // 2

    kernel_assert(len(padding) % 2 == 0, "padding length must be even, got " + str(len(padding)))
    kernel_assert(1 <= n_spatial <= 3, "padding must specify 1-3 spatial dims, got " + str(n_spatial))
    kernel_assert(ndim >= n_spatial, "input must have at least " + str(n_spatial) + " dims, got " + str(ndim))

    # Convert PyTorch padding order (innermost-first) to per-dim (D, H, W)
    pad_3d = [(0, 0), (0, 0), (0, 0)]
    for i in range(n_spatial):
        # PyTorch: padding[0:1] is innermost → pad_3d[2] (W), pad_3d[1] (H), pad_3d[0] (D)
        pad_3d[2 - i] = (padding[2 * i], padding[2 * i + 1])
    params = PadParams(pad=tuple(pad_3d), mode=mode)

    # Mode-specific padding constraints
    spatial_dims = x_ref.shape[ndim - n_spatial :]
    for i in range(n_spatial):
        dim_size = spatial_dims[n_spatial - 1 - i]
        pad_lo, pad_hi = padding[2 * i], padding[2 * i + 1]
        if mode == "reflect":
            kernel_assert(dim_size >= 2, "reflect requires size >= 2, got " + str(dim_size))
            kernel_assert(pad_lo < dim_size, "reflect padding must be < dim size")
            kernel_assert(pad_hi < dim_size, "reflect padding must be < dim size")
        elif mode == "circular":
            kernel_assert(pad_lo <= dim_size, "circular padding must be <= dim size")
            kernel_assert(pad_hi <= dim_size, "circular padding must be <= dim size")

    # Collapse batch dims into NC, pad spatial to 3 dims with leading 1s
    batch_dims = x_ref.shape[: ndim - n_spatial]

    NC = 1
    for b in batch_dims:
        NC *= b

    if n_spatial == 1:
        D, H, W = 1, 1, spatial_dims[0]
    elif n_spatial == 2:
        D, H, W = 1, spatial_dims[0], spatial_dims[1]
    else:
        D, H, W = spatial_dims[0], spatial_dims[1], spatial_dims[2]

    x_4d = x_ref.reshape((NC, D, H, W))
    out_4d = out_ref.reshape((NC, D + params.total(0), H + params.total(1), W + params.total(2)))

    return x_4d, out_4d, params


def _compute_output_shape(in_shape: tuple, padding: tuple) -> tuple:
    """Compute the output shape after padding (PyTorch convention)."""
    out = list(in_shape)
    n_spatial = len(padding) // 2
    for i in range(n_spatial):
        dim = len(in_shape) - 1 - i
        out[dim] += padding[2 * i] + padding[2 * i + 1]
    return tuple(out)


# ---------------------------------------------------------------------------
# Tile slicing
# ---------------------------------------------------------------------------


def _tile_input(x: nl.NkiTensor, dim: int, tile_strategy: PadTilingStrategy, tile_idx: int) -> nl.NkiTensor:
    """Slice input NkiTensor for tile *tile_idx* along spatial *dim*."""
    axis = dim + 1
    start = tile_idx * tile_strategy.tile_sizes[dim]
    end = min(start + tile_strategy.tile_sizes[dim], tile_strategy.src_sizes[dim])
    return x.slice(axis, start, end)


def _tile_output(
    out: nl.NkiTensor, dim: int, tile_strategy: PadTilingStrategy, tile_params: PadParams, tile_idx: int
) -> nl.NkiTensor:
    """Slice output NkiTensor for tile *tile_idx* along spatial *dim*.

    The output slice includes the tile's interior plus any SBUF padding
    assigned to this tile by ``tile_params``.
    """
    axis = dim + 1
    tile_start = tile_idx * tile_strategy.tile_sizes[dim]
    tile_count = min(tile_strategy.tile_sizes[dim], tile_strategy.src_sizes[dim] - tile_start)
    out_start = tile_strategy.params.before(dim) + tile_start - tile_params.before(dim)
    out_size = tile_params.before(dim) + tile_count + tile_params.after(dim)
    return out.slice(axis, out_start, out_start + out_size)


# ---------------------------------------------------------------------------
# Deferred DMA padding
# ---------------------------------------------------------------------------


def _fill_deferred_padding(
    out_view: nl.NkiTensor, pad_mode: PadMode, dim: int, tile_strategy: PadTilingStrategy
) -> None:
    """Fill padding on *dim* by copying from already-written output slices.

    Called after all tiles along *dim* have been processed, so the interior
    region of the output is fully populated (including inner-dimension padding).
    """
    axis = dim + 1
    pad_before = tile_strategy.params.before(dim)
    pad_after = tile_strategy.params.after(dim)
    src_count = tile_strategy.src_sizes[dim]

    for i in range(pad_before):
        src_idx = pad_mode.map_idx(dim, -(pad_before - i))
        dst = out_view.select(axis, i)
        src = out_view.select(axis, pad_before + src_idx)
        pad_mode.fill_deferred(dst, src)

    for i in range(pad_after):
        src_idx = pad_mode.map_idx(dim, i)
        dst = out_view.select(axis, pad_before + src_count + i)
        src = out_view.select(axis, pad_before + src_idx)
        pad_mode.fill_deferred(dst, src)


# ---------------------------------------------------------------------------
# Top-level kernel
# ---------------------------------------------------------------------------


@nki.jit
def pad(x_ref, padding, mode="replicate", value=0):
    """Pad a tensor with constant, replicate, reflect, or circular mode.

    Equivalent to ``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

    Args:
        x_ref: Input tensor (any number of batch dims + up to 3 spatial dims).
        padding: Padding amounts in PyTorch convention (innermost-first).
        mode: ``"constant"``, ``"replicate"``, ``"reflect"``, or ``"circular"``.
        value: Fill value for constant mode (default 0, ignored for other modes).

    Returns:
        Padded output tensor allocated in shared HBM.
    """
    dtype = x_ref.dtype
    out_shape = _compute_output_shape(x_ref.shape, padding)
    out_ref = nl.ndarray(out_shape, dtype=dtype, buffer=nl.shared_hbm)

    x_4d, out_4d, params = _normalize_to_4d(x_ref, out_ref, padding, mode)

    x_4d, out_4d, params = shard_operation(x_4d, out_4d, params, mode, dtype)
    if x_4d is None:
        return out_ref
    NC, D, H, W = x_4d.shape

    tile_strategy = compute_tiling_strategy(D, H, W, params, mode, dtype)
    TILE_P = nl.tile_size.pmax

    for nc_idx in nl.affine_range(div_ceil(NC, TILE_P)):
        nc_base = nc_idx * TILE_P
        nc_count = min(TILE_P, NC - nc_base)
        x_nc = x_4d.slice(0, nc_base, nc_base + nc_count)
        out_nc = out_4d.slice(0, nc_base, nc_base + nc_count)

        pad_mode = make_pad_mode(mode, x_nc, value=value)

        # --- Tiling loops: D → H → W ---
        for di in nl.affine_range(tile_strategy.n_tiles[0]):
            params_d = tile_strategy.tile_params(tile_strategy.params, 0, di)
            x_d = _tile_input(x_nc, 0, tile_strategy, di)
            out_d = _tile_output(out_nc, 0, tile_strategy, params_d, di)

            for hi in nl.affine_range(tile_strategy.n_tiles[1]):
                params_h = tile_strategy.tile_params(params_d, 1, hi)
                x_h = _tile_input(x_d, 1, tile_strategy, hi)
                out_h = _tile_output(out_d, 1, tile_strategy, params_h, hi)

                for wi in nl.affine_range(tile_strategy.n_tiles[2]):
                    params_w = tile_strategy.tile_params(params_h, 2, wi)
                    x_w = _tile_input(x_h, 2, tile_strategy, wi)
                    out_w = _tile_output(out_h, 2, tile_strategy, params_w, wi)

                    # Load contiguous input tile into SBUF
                    src_sb = nl.ndarray(x_w.shape, dtype=dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=src_sb, src=x_w)

                    # Pad in SBUF and store to HBM
                    padded_sb = nl.ndarray(out_w.shape, dtype=dtype, buffer=nl.sbuf)
                    pad_compute(src_sb, padded_sb, params_w, pad_mode)
                    nisa.dma_copy(dst=out_w, src=padded_sb)

                # Deferred W padding (reflect/circular on multi-tiled W)
                if tile_strategy.needs_deferred_padding(2):
                    _fill_deferred_padding(out_h, pad_mode, 2, tile_strategy)

            # Deferred H padding
            if tile_strategy.needs_deferred_padding(1):
                _fill_deferred_padding(out_d, pad_mode, 1, tile_strategy)

        # Deferred D padding
        if tile_strategy.needs_deferred_padding(0):
            _fill_deferred_padding(out_nc, pad_mode, 0, tile_strategy)

    return out_ref
