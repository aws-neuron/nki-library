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

"""Broadcast primitive: replicate partition rows."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.tensor_view import TensorView
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class Broadcast(nl.NKIObject):
    """Broadcast from single partition row to multiple rows."""

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
        src_partition: int = 0,
    ) -> None:
        self._dst = dst
        self._src = src
        self._src_partition = src_partition
        self._name = f"Broadcast(dst={dst.get_name()}, src={src.get_name()})"

    def execute(self) -> None:
        # Single-tile only: src is one partition row and dst is <= pmax rows. For a
        # larger dst, broadcast once into a <= pmax buffer and reuse it across tiles.
        kernel_assert(
            self._src.get_num_tiles() == 1 and self._dst.get_num_tiles() == 1,
            f"Broadcast '{self._name}': src and dst must each be a single tile",
        )

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()

        for _ in range(self._dst.get_num_tiles()):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()

            dst_npar = dst_tile.shape[0]
            # The src view below is sliced down to the single chosen partition row, so
            # within that view the source row is at index 0. shuffle_mask[i] selects the
            # source partition (within the 32-partition quadrant) that output row i copies
            # from -- it must therefore be 0, not src_partition (the slice already applied
            # that offset; putting src_partition here too would double-apply it).
            shuffle_mask = [0] * 32
            for i in range((dst_npar + 31) // 32):
                cur_npar = min(32, dst_npar - i * 32)
                nisa.nc_stream_shuffle(
                    src=src_tile.slice(dim=0, start=self._src_partition, end=self._src_partition + 1),
                    dst=dst_tile.slice(dim=0, start=i * 32, end=i * 32 + cur_npar),
                    shuffle_mask=shuffle_mask,
                )

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()


def broadcast(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
    src_partition: int = 0,
) -> None:
    """Compact broadcast: replicate partition row to all rows. Whole tensor, no tiling.

    Args:
        dst: Destination tensor in SBUF (multiple partition rows)
        src: Source tensor in SBUF (single partition row to broadcast)
        src_partition: Which partition row to broadcast (default 0)
    """
    dst_ts = tile_stream.tile(dst, get_logical_shape(dst), iter_order=RowMajor())
    src_ts = tile_stream.tile(src, get_logical_shape(src), iter_order=RowMajor())
    Broadcast(dst=dst_ts, src=src_ts, src_partition=src_partition).execute()
