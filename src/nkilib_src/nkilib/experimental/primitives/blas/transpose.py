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

"""
NKI Primitives Transpose Module.

PE-based transpose using nc_transpose (SBUF → PSUM → SBUF).
For DMA-based transpose during load/store, use dma.Load/Store with layout parameter.
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.logging import get_logger
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape, tile


class Transpose(nl.NKIObject):
    """PE transpose: SBUF → PSUM → SBUF via nc_transpose.

    Transposes the P and F dimensions of each tile.
    src tile (P, F) → dst tile (F', P') where F becomes new P and P becomes new F.

    Supports implicit broadcasting:
        - src (P, 1) → dst (B, P): broadcasts F dim from 1 to B
        - src (1, F) → dst (F, B): broadcasts P dim from 1 to B

    If src and dst have different dtypes, activation is used to cast.
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
    ) -> None:
        self._dst = dst
        self._src = src
        self._name = f"Transpose(src={src.get_name()}, dst={dst.get_name()})"
        self._logger = get_logger(self._name)

        src_tile = self._src.get_tile_shape()
        dst_tile = self._dst.get_tile_shape()

        kernel_assert(
            len(src_tile) == 2 and len(dst_tile) == 2,
            f"blas.Transpose '{self._name}' currently only supports 2D tiles, got src_tile={src_tile}, dst_tile={dst_tile}",
        )

        src_P, src_F = src_tile
        dst_P, dst_F = dst_tile

        # Determine broadcast factors
        # After transpose: src (P, F) → (F, P)
        # dst expects (dst_P, dst_F)
        # So: dst_P should match src_F (or src_F=1 for broadcast)
        #     dst_F should match src_P (or src_P=1 for broadcast)
        self._broadcast_P = 1  # Broadcast factor for src_F → dst_P
        self._broadcast_F = 1  # Broadcast factor for src_P → dst_F

        if src_F == 1 and dst_P > 1:
            # Broadcast F dimension: src (P, 1) → dst (B, P)
            self._broadcast_P = dst_P
            kernel_assert(
                src_P == dst_F,
                f"blas.Transpose '{self._name}' with F-broadcast expects src_P == dst_F, got src_P={src_P}, dst_F={dst_F}",
            )
        elif src_P == 1 and dst_F > 1:
            # Broadcast P dimension: src (1, F) → dst (F, B)
            self._broadcast_F = dst_F
            kernel_assert(
                src_F == dst_P,
                f"blas.Transpose '{self._name}' with P-broadcast expects src_F == dst_P, got src_F={src_F}, dst_P={dst_P}",
            )
        else:
            # No broadcast - standard transpose
            kernel_assert(
                src_P == dst_F and src_F == dst_P,
                f"blas.Transpose '{self._name}' expects dst tile to be transposed src tile, got src=(P={src_P}, F={src_F}), dst=(P={dst_P}, F={dst_F})",
            )

        # Validate number of tiles match
        src_num_tiles = self._src.get_num_tiles()
        dst_num_tiles = self._dst.get_num_tiles()
        kernel_assert(
            src_num_tiles == dst_num_tiles,
            f"blas.Transpose '{self._name}' expects src and dst to have same number of tiles, got src={src_num_tiles}, dst={dst_num_tiles}",
        )

        self._num_tiles = src_num_tiles
        self._src_P = src_P
        self._src_F = src_F
        self._dst_P = dst_P
        self._dst_F = dst_F
        self._needs_cast = self._src.get_dtype() != self._dst.get_dtype()
        self._needs_broadcast = self._broadcast_P > 1 or self._broadcast_F > 1

    def execute(self):
        """Transpose all tiles from src to dst."""
        for _ in range(self._num_tiles):
            src_tile = self._src.get_tile()
            dst_tile = self._dst.get_tile()

            # nc_transpose outputs to PSUM with shape (dst_P, dst_F)
            psum_tmp = nl.ndarray((self._dst_P, self._dst_F), dtype=self._src.get_dtype(), buffer=nl.psum)

            if self._needs_broadcast:
                if self._broadcast_P > 1:
                    # Broadcast F→P: src (P, 1) → dst (B, P)
                    # Use access pattern to repeat the single F value B times
                    nisa.nc_transpose(
                        dst=psum_tmp,
                        data=src_tile.ap(
                            pattern=[[self._src_P, self._src_F], [0, self._broadcast_P]],
                            offset=0,
                        ),
                    )
                else:
                    # Broadcast P→F: src (1, F) → dst (F, B)
                    # Use access pattern to repeat the single P value B times
                    nisa.nc_transpose(
                        dst=psum_tmp,
                        data=src_tile.ap(
                            pattern=[[self._src_P, self._src_F], [self._broadcast_F, 0]],
                            offset=0,
                        ),
                    )
            else:
                # Standard transpose
                nisa.nc_transpose(dst=psum_tmp, data=src_tile)

            # Copy from PSUM to SBUF dst (with cast if dtypes differ)
            if self._needs_cast:
                nisa.activation(dst=dst_tile, op=nl.copy, data=psum_tmp)
            else:
                nisa.tensor_copy(dst=dst_tile, src=psum_tmp)

        self._src.reset_cur_tile()
        self._dst.reset_cur_tile()

    def get_name(self) -> str:
        return self._name


def transpose(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
    src_has_p_tile_dim: bool = True,
    dst_has_p_tile_dim: bool = True,
) -> None:
    """Compact transpose: single-tile nc_transpose with optional dtype cast.

    Wraps src and dst in single-tile TileStreams and calls Transpose.execute().

    Args:
        dst: Destination tensor in SBUF (P', F') where P'=src_F, F'=src_P
        src: Source tensor in SBUF (P, F)
        src_has_p_tile_dim: If True (default), src is an alloc_logical container.
            If False, src is a raw nl.NkiTensor without p_tile dim.
        dst_has_p_tile_dim: If True (default), dst is an alloc_logical container.
            If False, dst is a raw nl.NkiTensor without p_tile dim.
    """
    src_shape = get_logical_shape(src) if src_has_p_tile_dim else tuple(src.shape)
    dst_shape = get_logical_shape(dst) if dst_has_p_tile_dim else tuple(dst.shape)
    src_ts = tile(src, src_shape, iter_order=RowMajor(), has_p_tile_dim=src_has_p_tile_dim)
    dst_ts = tile(dst, dst_shape, iter_order=RowMajor(), has_p_tile_dim=dst_has_p_tile_dim)
    Transpose(dst=dst_ts, src=src_ts).execute()
