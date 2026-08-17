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

"""Generalized matmul loop nest abstraction for nc_matmul.

Handles the K × M × N inner loop with callbacks for weight loading,
post-matmul operations, and drain.
"""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.tiled_tensor import TiledTensor as _CoreTiledTensor


class _DimsAccessor(nl.NKIObject):
    """Wraps an nD TiledTensor with dim specification into a 2D (K, X) accessor."""

    def __init__(self, tt, k_d, x_d, k_tiles, x_tiles):
        self._tt = tt
        self._k_dim = k_d
        self._x_dim = x_d
        self.shape = (k_tiles, x_tiles)
        # Compute effective tile_size: (K_tile, X_tile)
        # K is the partition dim (first non-1 or first dim), X is the free dim (last non-1 or last dim)
        ts = tt.tile_size
        # K tile size: the partition dim tile size (first element or first non-1)
        k_ts = ts[0] if len(ts) > 0 else 1
        # X tile size: the free dim tile size (last element)
        x_ts = ts[-1] if len(ts) > 0 else 1
        self.tile_size = (k_ts, x_ts)

    def __getitem__(self, idx):
        k, x = idx
        nd = len(self._tt._nd_grid) if hasattr(self._tt, '_nd_grid') else 2
        if nd == 2:
            if hasattr(self._tt, '_tiles') and self._tt._tiles is not None:
                return self._tt[k, x]
            return self._tt.get_tile((k, x), keep_dim=False)
        indices = [0] * nd
        if self._k_dim is not None:
            indices[self._k_dim] = k
        if self._x_dim is not None:
            indices[self._x_dim] = x
        idx_tuple = []
        for d in range(nd):
            ts = self._tt._nd_tile_size[d]
            if ts == 1:
                idx_tuple.append(indices[d])
            else:
                start = indices[d] * ts
                actual_dim = (
                    self._tt._nd_actual_shape[d] if hasattr(self._tt, '_nd_actual_shape') else self._tt.tensor.shape[d]
                )
                end = min(start + ts, actual_dim)
                idx_tuple.append(slice(start, end))
        return self._tt.tensor[tuple(idx_tuple)]


def _resolve_dims(tensor, dims):
    """Wrap an nD TiledTensor with dim specification into a 2D (K, X) accessor.

    Args:
        tensor: TiledTensor (2D or nD)
        dims: dict mapping role → dim index, e.g. {"K": 1, "N": 2}.
              If None, tensor is already 2D (K, M) or (K, N).

    Returns:
        (accessor, K_tiles, X_tiles) where accessor(k, x) returns a 2D tile.
    """
    if dims is None:
        return tensor, tensor.shape[0], tensor.shape[1]

    k_dim = dims.get("K")
    other_dim = dims.get("M")
    if other_dim is None:
        other_dim = dims.get("N")

    grid = tensor._nd_grid if hasattr(tensor, '_nd_grid') else tensor.shape
    K_tiles = grid[k_dim] if k_dim is not None else 1
    X_tiles = grid[other_dim] if other_dim is not None else 1

    return _DimsAccessor(tensor, k_dim, other_dim, K_tiles, X_tiles), K_tiles, X_tiles


def matmul_loop_nest(
    stationary,
    moving,
    dst_psum=None,
    dst_sbuf=None,
    output=None,
    auxiliaries=None,
    on_output=None,
    col_factor=1,
    col_dim=128,
    n_packing=1,
    num_psum_banks=8,
    on_n_start=None,
    on_n_end=None,
    load_weights=None,
    on_post_matmul=None,
    on_accumulate=None,
    on_k_group_end=None,
    on_drain=None,
    should_compute=None,
    matmul_kwargs=None,
    loop_order="NM",
    stationary_dims=None,
    moving_dims=None,
):
    """Matmul inner loop nest: N × K × M.

    Computes output[M, N] = stationary[K, M].T @ moving[K, N].

    Args:
        stationary: TiledTensor [K, M] in SBUF — stationary operand
        moving: TiledTensor [K, N] in SBUF — moving operand (rotate_dim=0 for rotating buffers)
        dst_psum: TiledTensor [M, N] in PSUM — matmul output. None = auto-allocate.
        dst_sbuf: TiledTensor [M, N] in SBUF — drain target. None = no drain.
        col_factor: PE column tiling factor (1, 2, or 4)
        col_dim: PE column tile size
        n_packing: Pack this many N tiles into one PSUM bank at different column
                   offsets. Drain fires once per group with the full bank.
                   Use when N_tile < 512 to maximize PSUM utilization.
        num_psum_banks: PSUM banks for auto-allocation
        on_n_start: fn(n) — called at start of each N tile
        on_n_end: fn(n) — called at end of each N tile
        load_weights: fn(k, n, buf) — load weights for tile (k, n) into moving buffer
        on_post_matmul: fn(psum_tile, k, m, n) — per-tile callback inside K loop
        on_k_group_end: fn(psum_tile, m, n, col_factor) — reduce PE column slots
        on_drain: fn(psum_tile, sbuf_tile, m, n) — custom drain after all K tiles
        should_compute: fn(k, m, n) → bool — skip matmul when False
        matmul_kwargs: fn(k, col_idx) → dict — extra nc_matmul args
        loop_order: "NM" (default) or "MN" — outer/inner loop order
        stationary_dims: dict mapping role → dim index for nD stationary,
                         e.g. {"K": 1, "M": 2}. None = already 2D (K, M).
        moving_dims: dict mapping role → dim index for nD moving,
                     e.g. {"K": 1, "N": 2}. None = already 2D (K, N).

    Parser mode: Callbacks must be module-level functions or bound methods
    of an NKIObject. Use bound methods to pass state without closures:

        @dataclass
        class MyCb(nl.NKIObject):
            bias: object
            def on_drain(self, psum_tile, sbuf_tile, m, n):
                nisa.tensor_tensor(dst=sbuf_tile, data1=psum_tile, data2=self.bias, op=nl.add)

        cb = MyCb(bias=bias_sb)
        matmul_loop_nest(..., on_drain=cb.on_drain)
    """
    # v2 compatibility: map new params to old
    _on_output = on_output
    _auxiliaries = auxiliaries if auxiliaries is not None else []
    _output = output
    if output is not None and dst_sbuf is None:
        dst_sbuf = output
    if on_accumulate is not None and on_post_matmul is None:
        on_post_matmul = on_accumulate
    if _on_output is not None and on_drain is None and n_packing <= 1:
        # Simple case: no packing, wrap on_output as on_drain
        def _v2_drain(psum_tile, sbuf_tile, m, n):
            aux_views = [
                aux.get_tile((m, n), keep_dim=True) if hasattr(aux, 'get_tile') else aux[m, n] for aux in _auxiliaries
            ]
            _on_output(psum_tile, sbuf_tile, *aux_views)

        on_drain = _v2_drain

    # Resolve nD dims to 2D accessors
    _stat_has_n = False
    _mov_has_m = False
    if stationary_dims is not None:
        stationary, K_tiles_s, X_tiles_s = _resolve_dims(stationary, stationary_dims)
        if "N" in stationary_dims and stationary_dims.get("N") is not None:
            _stat_has_n = True
            M_tiles = 1
        else:
            M_tiles = X_tiles_s
    else:
        K_tiles_s = stationary.shape[0]
        M_tiles = stationary.shape[1]

    if moving_dims is not None:
        moving, K_tiles_m, X_tiles_m = _resolve_dims(moving, moving_dims)
        if "M" in moving_dims and moving_dims.get("M") is not None:
            _mov_has_m = True
    else:
        K_tiles_m = moving.shape[0]

    K_tiles = max(K_tiles_s if stationary_dims else stationary.shape[0], K_tiles_m if moving_dims else moving.shape[0])

    # Auto-allocate PSUM
    if dst_psum is None:
        M_tile_sz = stationary.tile_size[1]
        N_tile_sz = moving.tile_size[1]
        psum_n_sz = N_tile_sz * n_packing
        N_psum_tiles = num_psum_banks
        total_psum_banks = M_tiles * N_psum_tiles
        dst_psum = _CoreTiledTensor.alloc(
            grid=(M_tiles, N_psum_tiles),
            tile_size=(M_tile_sz, psum_n_sz),
            dtype=nl.float32,
            buffer=nl.psum,
            num_banks=total_psum_banks,
        )

    # Detect factored-N: stat and mov have independent N dimensions
    _N0 = 1  # stat's N dim (h2)
    _N1 = 1  # mov's N dim (bxs_tiles)
    _factored_n = False
    if _stat_has_n and moving_dims is not None and "K" in moving_dims:
        _N0 = X_tiles_s
        # Get mov's non-K dim size from the original tensor (before _resolve_dims wrapping)
        _mov_orig_shape = moving._tt.shape if hasattr(moving, '_tt') else moving.shape
        _N1_dim = moving_dims.get("N") if moving_dims.get("N") is not None else None
        if _N1_dim is not None:
            _N1 = _mov_orig_shape[_N1_dim]
        else:
            # mov only has K specified — check if its grid has a non-K dim > 1
            for d in range(len(_mov_orig_shape)):
                if d != moving_dims.get("K") and _mov_orig_shape[d] > 1:
                    _N1 = _mov_orig_shape[d]
                    break
        if _N0 > 1 and _N1 > 1:
            _factored_n = True

    N_tiles = (
        dst_sbuf.shape[1]
        if (dst_sbuf and n_packing == 1)
        else (dst_psum.shape[1] * n_packing if n_packing > 1 else dst_psum.shape[1])
    )
    if _factored_n and not dst_sbuf:
        N_tiles = _N0 * _N1
    if _on_output is not None and _factored_n and _N1 > 1:
        N_tiles = _N0 * _N1
    if _on_output is not None and n_packing > 1 and _output is not None:
        N_tiles = _output.shape[1] * n_packing
    # When stat has N dim, use actual N count (not padded by n_packing)
    if _stat_has_n and n_packing > 1 and not _factored_n:
        N_tiles = X_tiles_s
    N_tile_sz_for_packing = moving.tile_size[-1] if n_packing > 1 else 0

    # K-outermost mode: K → M → N (or K → N → M), drain after all K
    if loop_order in ("KMN", "KNM"):
        for k in nl.affine_range(K_tiles):
            if load_weights:
                load_weights(k, 0, moving[k, 0])
            if loop_order == "KMN":
                outer_k, inner_k = M_tiles, N_tiles
            else:
                outer_k, inner_k = N_tiles, M_tiles
            for outer_i in nl.affine_range(outer_k):
                for inner_i in nl.affine_range(inner_k):
                    m = outer_i if loop_order == "KMN" else inner_i
                    n = inner_i if loop_order == "KMN" else outer_i
                    stat = stationary[k, m]
                    mov = moving[k, n]
                    psum = dst_psum[m, n]
                    nisa.nc_matmul(dst=psum, stationary=stat, moving=mov)
        # Drain all (m, n) after K loop completes
        for m in nl.affine_range(M_tiles):
            for n in nl.affine_range(N_tiles):
                psum = dst_psum[m, n]
                if _on_output is not None:
                    out_tile = _output[m, n] if _output is not None else None
                    aux_views = [
                        aux.get_tile((m, n), keep_dim=True) if hasattr(aux, 'get_tile') else aux[m, n]
                        for aux in _auxiliaries
                    ]
                    _on_output(psum, out_tile, *aux_views)
                elif on_drain:
                    on_drain(psum, dst_sbuf[m, n] if dst_sbuf else None, m, n)
                elif dst_sbuf is not None:
                    nisa.tensor_copy(dst=dst_sbuf[m, n], src=psum)
        return

    if loop_order == "NM":
        outer_range, inner_range = N_tiles, M_tiles
    else:
        outer_range, inner_range = M_tiles, N_tiles

    # With n_packing, the outer N loop iterates in groups
    if n_packing > 1 and loop_order == "NM":
        n_groups = (N_tiles + n_packing - 1) // n_packing
        outer_range = n_groups

    for outer in nl.affine_range(outer_range):
        m_outer, n_outer = (None, outer) if loop_order == "NM" else (outer, None)

        if on_n_start and loop_order == "NM":
            on_n_start(outer)

        for k_group_start in nl.affine_range(0, K_tiles, col_factor):
            for col_idx in range(min(col_factor, K_tiles - k_group_start)):
                k = k_group_start + col_idx

                if load_weights:
                    n_for_load = outer if loop_order == "NM" else 0
                    load_weights(k, n_for_load, moving[k, n_for_load])

                # N-packing: iterate over packed N tiles within this group
                n_pack_range = n_packing if (n_packing > 1 and loop_order == "NM") else 1
                for n_pack_idx in range(n_pack_range):
                    for inner in nl.affine_range(inner_range):
                        if n_packing > 1 and loop_order == "NM":
                            m = inner
                            n = outer * n_packing + n_pack_idx
                        else:
                            m = inner if loop_order == "NM" else outer
                            n = outer if loop_order == "NM" else inner

                        if n >= N_tiles:
                            continue

                        if should_compute and not should_compute(k, m, n):
                            continue

                        # Access tiles — stationary may use n instead of m if it has N dim
                        stat_idx2 = n if _stat_has_n else m
                        mov_idx2 = m if _mov_has_m else n
                        if _factored_n:
                            stat = stationary[k, n // _N1]
                            mov = moving[k % moving.shape[0], n % _N1]
                        else:
                            stat = stationary[k, stat_idx2 % stationary.shape[1]]
                            mov = moving[k % moving.shape[0], mov_idx2 % moving.shape[1]]

                        # PSUM: with n_packing, index by group not individual n
                        if n_packing > 1 and loop_order == "NM":
                            psum = dst_psum[m, outer]
                        else:
                            psum = dst_psum[m, n]

                        kwargs = matmul_kwargs(k, col_idx) if matmul_kwargs else None

                        # Auto-clamp K (partition) dim to min of stat and mov
                        K_actual = min(stat.shape[0], mov.shape[0])
                        if stat.shape[0] != K_actual:
                            stat = stat[:K_actual, :]
                        if mov.shape[0] != K_actual:
                            mov = mov[:K_actual, :]

                        # Slice PSUM to match matmul output shape
                        M_actual = stat.shape[-1]
                        N_actual = mov.shape[-1]
                        if col_factor > 1:
                            psum_slice = psum[nl.ds(col_dim * col_idx, M_actual), :N_actual]
                        elif n_packing > 1:
                            # N-packing: offset within the bank
                            n_col_start = n_pack_idx * N_tile_sz_for_packing
                            psum_slice = psum[:M_actual, nl.ds(n_col_start, N_actual)]
                        elif psum.shape[0] != M_actual or psum.shape[1] != N_actual:
                            psum_slice = psum[:M_actual, :N_actual]
                        else:
                            psum_slice = psum

                        if kwargs is not None:
                            nisa.nc_matmul(
                                dst=psum_slice,
                                stationary=stat,
                                moving=mov,
                                perf_mode=kwargs.get("perf_mode"),
                                tile_position=kwargs.get("tile_position"),
                                tile_size=kwargs.get("tile_size"),
                                is_transpose=kwargs.get("is_transpose"),
                            )
                        else:
                            nisa.nc_matmul(dst=psum_slice, stationary=stat, moving=mov)

                        if on_post_matmul:
                            on_post_matmul(psum_slice, k, m, n)

            if col_factor > 1 and on_k_group_end:
                for inner in nl.affine_range(inner_range):
                    m = inner if loop_order == "NM" else outer
                    n = outer if loop_order == "NM" else inner
                    on_k_group_end(dst_psum[m, n], m, n, col_factor)

        # Drain after all K tiles (and all packed N tiles)
        for inner in nl.affine_range(inner_range):
            if n_packing > 1 and loop_order == "NM":
                m = inner
                psum_bank = dst_psum[m, outer]
                if _on_output is not None:
                    # v2: fire once per bank, clamp PSUM to output size
                    out_tile = _output[m, outer] if _output is not None else None
                    if out_tile is not None and out_tile.shape[-1] < psum_bank.shape[-1]:
                        psum_bank = psum_bank[:, : out_tile.shape[-1]]
                    aux_views = [aux[m, outer] for aux in _auxiliaries]
                    _on_output(psum_bank, out_tile, *aux_views)
                elif on_drain:
                    on_drain(psum_bank, dst_sbuf[m, outer] if dst_sbuf else None, m, outer)
                elif dst_sbuf is not None:
                    nisa.tensor_copy(dst=dst_sbuf[m, outer], src=psum_bank)
            else:
                m = inner if loop_order == "NM" else outer
                n = outer if loop_order == "NM" else inner
                psum = dst_psum[m, n]
                # Always slice PSUM — PSUM banks may be larger than actual output
                stat_idx2 = n if _stat_has_n else m
                mov_idx2 = m if _mov_has_m else n
                if _factored_n:
                    stat_for_drain = stationary[0, n // _N1]
                    mov_for_drain = moving[0, n % _N1]
                else:
                    stat_for_drain = stationary[0, stat_idx2 % stationary.shape[1]]
                    mov_for_drain = moving[0 % moving.shape[0], mov_idx2 % moving.shape[1]]
                psum = psum[: stat_for_drain.shape[-1], : mov_for_drain.shape[-1]]
                if _on_output is not None and _factored_n and _N1 > 1:
                    # Fire on_output per bxs_tile with correct output slice
                    n0 = n // _N1  # h2 index
                    n1 = n % _N1  # bxs_tile index
                    out_full = _output[m, n0]
                    bxs_col = n1 * mov_for_drain.shape[-1]
                    actual_cols = min(mov_for_drain.shape[-1], out_full.shape[-1] - bxs_col)
                    out_slice = out_full.slice(dim=1, start=bxs_col, end=bxs_col + actual_cols)
                    psum_clamped = psum[: stat_for_drain.shape[-1], :actual_cols]
                    aux_views = [
                        aux.get_tile((m, n0), keep_dim=True) if hasattr(aux, 'get_tile') else aux[m, n0]
                        for aux in _auxiliaries
                    ]
                    _on_output(psum_clamped, out_slice, *aux_views)
                elif on_drain:
                    on_drain(psum, dst_sbuf[m, n] if dst_sbuf else None, m, n)
                elif dst_sbuf is not None:
                    nisa.tensor_copy(dst=dst_sbuf[m, n], src=psum)

        if on_n_end and loop_order == "NM":
            on_n_end(outer)
