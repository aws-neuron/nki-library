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

"""Matmul primitive for nkiprimitives."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa import matmul_perf_mode

from ....core.utils.allocator import sizeinbytes
from ....core.utils.common_types import QuantizationType
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, reduce
from ..tile_stream import TileStream
from ..view_spec import Broadcast, Permute, ReshapeDim, Select, Slice, ViewSpec

F_MAX = 512


def _get_k_position(ts: TileStream) -> int:
    """Get the position of K (first tiled dim) in the grid (logical dimension order)."""
    return ts.get_num_virtual() + ts.get_tile_dims()[0]


def _free_grid(grid, k_pos: int):
    """Product of all grid dims except K at position k_pos."""
    return reduce('mul', grid, 1) // grid[k_pos]


class Matmul(nl.NKIObject):
    """Matrix multiplication: dst = stationary^T @ moving (+ bias).

    Loop structure (dst-driven):
        for each dst_tile:
            for stat_idx in stationary_range:      # stat_per_output iterations
                for mov_idx in moving_range:       # mov_per_output iterations
                    for k in K:                    # reduction
                        matmul(stat[stat_idx][k], mov[mov_idx][k])
            evict_to_dst()

    Current modes (simple, single-dimension packing):
        - stationary_pack: stat_per_output = pack_factor, mov_per_output = 1
        - moving_pack: stat_per_output = 1, mov_per_output = pack_factor
        - no_pack: stat_per_output = 1, mov_per_output = 1

    Future: mixed packing where both > 1, with stat_per_output * mov_per_output <= pack_factor
    """

    def __init__(
        self,
        dst: TileStream,
        moving: TileStream,
        stationary: TileStream,
        moving_scale: Optional[TileStream] = None,
        stationary_scale: Optional[TileStream] = None,
        bias: Optional[TileStream] = None,
        psum_evict_view: Optional[ViewSpec] = None,
        dequant_scale: Optional[TileStream] = None,
        dequant_type: QuantizationType = QuantizationType.NONE,
        perf_mode: Optional[str] = None,
        psum_buffer_degree: Optional[int] = None,
        skip_evict: bool = False,
    ) -> None:
        self._dst = dst
        self._moving = moving
        self._stationary = stationary
        self._moving_scale = moving_scale
        self._stationary_scale = stationary_scale
        self._bias = bias
        self._psum_evict_view = psum_evict_view
        self._dequant_scale = dequant_scale
        self._dequant_type = dequant_type
        self._perf_mode = perf_mode
        self._psum_buffer_degree = psum_buffer_degree
        self._skip_evict = skip_evict

        if skip_evict:
            kernel_assert(
                dst.get_container().buffer == nl.psum,
                "skip_evict=True requires dst buffer to be nl.psum",
            )

        self._use_mx = moving_scale is not None and stationary_scale is not None
        kernel_assert(
            not (moving_scale is None and stationary_scale is not None),
            "nc_matmul_mx requires both scales; got stationary_scale without moving_scale",
        )
        kernel_assert(
            not (moving_scale is not None and stationary_scale is None),
            "nc_matmul_mx requires both scales; got moving_scale without stationary_scale",
        )

        self._use_dequant = dequant_type != QuantizationType.NONE
        if self._use_dequant:
            kernel_assert(dequant_scale is not None, "dequant_scale required")
        if dequant_scale is not None:
            kernel_assert(dequant_type != QuantizationType.NONE, "dequant_scale provided but dequant_type is NONE")

        dst_tile = dst.get_base_tile_shape()
        moving_tile = moving.get_base_tile_shape()
        stationary_tile = stationary.get_base_tile_shape()

        kernel_assert(moving_tile[-2] == stationary_tile[-2], "K tile mismatch")
        kernel_assert(dst_tile[-2] == stationary_tile[-1], "dst P mismatch with stationary F")
        kernel_assert(dst_tile[-1] % moving_tile[-1] == 0, "dst F must be divisible by moving F")

        self._p_tile = dst_tile[-2]
        self._f_tile = moving_tile[-1]
        self._dst_f_tile = dst_tile[-1]
        self._pack_factor = dst_tile[-1] // moving_tile[-1]

        moving_grid = moving.get_tile_grid()
        stationary_grid = stationary.get_tile_grid()
        moving_k_pos = _get_k_position(moving)
        stationary_k_pos = _get_k_position(stationary)

        self._k_grid = moving_grid[moving_k_pos]
        self._stationary_grid = _free_grid(stationary_grid, stationary_k_pos)
        self._moving_grid = _free_grid(moving_grid, moving_k_pos)

        kernel_assert(moving_grid[moving_k_pos] == stationary_grid[stationary_k_pos], "K grid mismatch")

        # Packing configuration: (stat_per_output, mov_per_output)
        # For now, simple single-dimension packing. Future: mixed packing.
        if self._pack_factor > 1 and self._stationary_grid > 1:
            self._stat_per_output = self._pack_factor
            self._mov_per_output = 1
        elif self._pack_factor > 1 and self._moving_grid > 1:
            self._stat_per_output = 1
            self._mov_per_output = self._pack_factor
        else:
            self._stat_per_output = 1
            self._mov_per_output = 1

        self._num_stat_groups = div_ceil(self._stationary_grid, self._stat_per_output)
        self._num_mov_groups = div_ceil(self._moving_grid, self._mov_per_output)

        # PSUM bank tracking for explicit allocation
        self._psum_idx = 0

    def _alloc_psum(self, p_size: int, f_size: int):
        """Allocate PSUM buffer with optional explicit bank addressing."""
        if self._psum_buffer_degree is None:
            # Auto allocation - let compiler decide
            return nl.ndarray((p_size, f_size), dtype=nl.float32, buffer=nl.psum)
        else:
            # Explicit PSUM bank allocation with rotation
            psum_bank_size = F_MAX * sizeinbytes(nl.float32)
            address = (0, (self._psum_idx % self._psum_buffer_degree) * psum_bank_size)
            self._psum_idx = (self._psum_idx + 1) % self._psum_buffer_degree
            return nl.ndarray((p_size, f_size), dtype=nl.float32, buffer=nl.psum, address=address)

    def execute(self) -> None:
        self._reset_streams()
        self._psum_idx = 0  # Reset PSUM index at start of execute

        # Cache auxiliary tiles
        dequant_cache = []
        if self._use_dequant:
            for _ in range(self._dequant_scale.get_num_tiles()):
                dequant_cache.append(self._dequant_scale.get_tile())

        bias_cache = []
        if self._bias is not None:
            for _ in range(self._bias.get_num_tiles()):
                bias_cache.append(self._bias.get_tile())

        # Cache all moving tiles: [moving_grid][k_grid]
        moving_cache = []
        for _ in range(self._moving_grid):
            k_tiles = []
            for _ in range(self._k_grid):
                k_tiles.append(self._moving.get_tile())
            moving_cache.append(k_tiles)

        moving_scale_cache = []
        if self._use_mx:
            for _ in range(self._moving_grid):
                k_tiles = []
                for _ in range(self._k_grid):
                    k_tiles.append(self._moving_scale.get_tile())
                moving_scale_cache.append(k_tiles)

        # Cache all stationary tiles: [stationary_grid][k_grid]
        stationary_cache = []
        for _ in range(self._stationary_grid):
            k_tiles = []
            for _ in range(self._k_grid):
                k_tiles.append(self._stationary.get_tile())
            stationary_cache.append(k_tiles)

        stationary_scale_cache = []
        if self._use_mx:
            for _ in range(self._stationary_grid):
                k_tiles = []
                for _ in range(self._k_grid):
                    k_tiles.append(self._stationary_scale.get_tile())
                stationary_scale_cache.append(k_tiles)

        dequant_idx = 0
        bias_idx = 0

        # Main loop: dst tiles drive iteration
        for stat_group in range(self._num_stat_groups):
            stat_start = stat_group * self._stat_per_output
            stat_count = min(self._stat_per_output, self._stationary_grid - stat_start)

            for mov_group in range(self._num_mov_groups):
                mov_start = mov_group * self._mov_per_output
                mov_count = min(self._mov_per_output, self._moving_grid - mov_start)

                # Choose accumulation target
                if self._skip_evict:
                    dst_tile = self._dst.get_tile()
                    accum = dst_tile
                else:
                    accum = self._alloc_psum(self._p_tile, self._dst_f_tile)

                # Track actual P size written to PSUM (from first stationary tile)
                actual_p_size = stationary_cache[stat_start][0].shape[-1]

                # Accumulate matmuls into PSUM
                for s_local in range(stat_count):
                    stat_idx = stat_start + s_local
                    for m_local in range(mov_count):
                        mov_idx = mov_start + m_local
                        f_offset = (s_local * mov_count + m_local) * self._f_tile

                        for k in range(self._k_grid):
                            actual_p_size = stationary_cache[stat_idx][k].shape[-1]
                            # Use actual F size from moving tile (handles partial last tiles)
                            actual_f = moving_cache[mov_idx][k].shape[-1]
                            psum_slice = accum[:actual_p_size, nl.ds(f_offset, actual_f)]
                            if self._use_mx:
                                nisa.nc_matmul_mx(
                                    dst=psum_slice,
                                    stationary=stationary_cache[stat_idx][k],
                                    moving=moving_cache[mov_idx][k],
                                    stationary_scale=stationary_scale_cache[stat_idx][k],
                                    moving_scale=moving_scale_cache[mov_idx][k],
                                )
                            elif self._perf_mode == "double_row":
                                nisa.nc_matmul(
                                    dst=psum_slice,
                                    stationary=stationary_cache[stat_idx][k],
                                    moving=moving_cache[mov_idx][k],
                                    perf_mode=matmul_perf_mode.double_row,
                                )
                            else:
                                nisa.nc_matmul(
                                    dst=psum_slice,
                                    stationary=stationary_cache[stat_idx][k],
                                    moving=moving_cache[mov_idx][k],
                                )

                # Evict PSUM to dst (skip when accumulating directly into dst PSUM)
                if not self._skip_evict:
                    last_mov_f = moving_cache[mov_start + mov_count - 1][0].shape[-1]
                    max_f_size = (stat_count * mov_count - 1) * self._f_tile + last_mov_f
                    dequant_idx, bias_idx = self._evict_psum(
                        accum, actual_p_size, max_f_size, dequant_cache, bias_cache, dequant_idx, bias_idx
                    )

        self._reset_streams()

    def _evict_psum(self, accum, actual_p_size, max_f_size, dequant_cache, bias_cache, dequant_idx, bias_idx):
        """Evict PSUM to dst, applying dequant and bias if configured."""
        # Get dst_tile to determine actual F size (handles partial F tiles)
        dst_tile = self._dst.get_tile()
        # Flat F size: product of all non-partition dims (tile_view may reshape F into multiple dims)
        dst_flat_f = reduce('mul', dst_tile.shape[1:], 1)
        actual_f_size = min(max_f_size, dst_flat_f)

        # Slice dst_tile to match actual_p_size written to PSUM
        dst_tile = dst_tile.slice(0, 0, actual_p_size)

        psum_view = accum.slice(0, 0, actual_p_size).slice(1, 0, actual_f_size)

        if self._psum_evict_view is not None:
            for op in self._psum_evict_view.get_ops():
                if isinstance(op, Broadcast):
                    dim_size = psum_view.shape[op.dim]
                    if dim_size == 1:
                        psum_view = psum_view.broadcast(op.dim, op.size)
                    else:
                        psum_view = psum_view.reshape_dim(op.dim, (dim_size, 1))
                        psum_view = psum_view.broadcast(op.dim + 1, op.size)
                elif isinstance(op, Slice):
                    psum_view = psum_view.slice(op.dim, op.start, op.end)
                elif isinstance(op, Select):
                    psum_view = psum_view.select(op.dim, op.index)
                elif isinstance(op, ReshapeDim):
                    psum_view = psum_view.reshape_dim(op.dim, op.shape)
                elif isinstance(op, Permute):
                    psum_view = psum_view.permute(op.dims)

        dst_view = dst_tile
        psum = psum_view

        if self._use_dequant:
            dequant_tile = dequant_cache[dequant_idx % len(dequant_cache)]
            dequant_idx = dequant_idx + 1

            if self._dequant_type == QuantizationType.ROW:
                nisa.tensor_tensor(dst=dst_view, data1=psum, data2=dequant_tile, op=nl.multiply)
            elif self._dequant_type == QuantizationType.STATIC:
                scale_container = self._dequant_scale.get_container()
                scale_base = scale_container
                if len(scale_base.shape) == 3 and scale_base.shape[1] == 1:
                    scale_base = scale_base[:, 0, :]
                p_size = psum.shape[0]
                nisa.activation(dst=dst_view, data=psum, op=nl.copy, scale=scale_base[:p_size, :])
            evicted = dst_view
        else:
            evicted = psum

        if self._bias is not None:
            bias_tile = bias_cache[bias_idx % len(bias_cache)]
            bias_idx = bias_idx + 1
            # Slice bias_tile to match actual_p_size
            bias_tile = bias_tile.slice(0, 0, actual_p_size)
            bias_view = bias_tile

            if len(bias_view.shape) == 3 and len(evicted.shape) == 2:
                dst_3d = dst_tile.reshape_dim(1, (bias_view.shape[1], bias_view.shape[2]))
                if self._use_dequant:
                    nisa.tensor_tensor(dst=dst_3d, data1=dst_3d, data2=bias_view, op=nl.add)
                else:
                    psum_3d = psum_view.reshape_dim(1, (bias_view.shape[1], bias_view.shape[2]))
                    nisa.tensor_tensor(dst=dst_3d, data1=psum_3d, data2=bias_view, op=nl.add)
            else:
                nisa.tensor_tensor(dst=dst_view, data1=evicted, data2=bias_view, op=nl.add)
        elif not self._use_dequant:
            nisa.tensor_copy(dst=dst_view, src=psum)

        return dequant_idx, bias_idx

    def _reset_streams(self):
        """Reset all TileStream cursors."""
        self._dst.reset_cur_tile()
        self._moving.reset_cur_tile()
        self._stationary.reset_cur_tile()
        if self._bias is not None:
            self._bias.reset_cur_tile()
        if self._use_mx:
            self._moving_scale.reset_cur_tile()
            self._stationary_scale.reset_cur_tile()
        if self._use_dequant:
            self._dequant_scale.reset_cur_tile()
