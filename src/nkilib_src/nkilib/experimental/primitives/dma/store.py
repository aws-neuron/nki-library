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

"""Store primitive: SBUF TileStream to HBM."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import HBMStream, TileStream, get_logical_shape, tile_hbm


class Store(nl.NKIObject):
    """Store data from TileStream in SBUF to HBM.

    dst is an HBMStream constructed by the user via tile_hbm().
    src is a TileStream constructed by the user via tile_stream.tile().
    """

    def __init__(
        self,
        dst: HBMStream,
        src: TileStream,
    ) -> None:
        self._src = src
        self._dst = dst
        self._name = f"Store(src={src.get_name()})"

        kernel_assert(
            src.get_num_tiles() == dst.get_num_tiles(),
            f"Store '{self._name}': tile count mismatch - src={src.get_num_tiles()}, dst={dst.get_num_tiles()}",
        )

    def execute(self) -> None:
        self._src.reset_cur_tile()
        self._dst.reset_cur_tile()

        for _ in range(self._dst.get_num_tiles()):
            src_tile = self._src.get_tile()
            dst_tile = self._dst.get_tile()
            nisa.dma_copy(dst=dst_tile, src=src_tile)

        self._src.reset_cur_tile()
        self._dst.reset_cur_tile()


def store(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
) -> None:
    """Compact store: SBUF to HBM. Whole tensor, no tiling.

    Wraps dst in a single-tile HBMStream and src in a single-tile TileStream.

    Args:
        dst: Destination tensor in HBM
        src: Source tensor in SBUF (from alloc_logical or nl.NkiTensor)
    """
    logical_shape = get_logical_shape(src)
    src_ts = tile_stream.tile(src, logical_shape, iter_order=RowMajor())

    dst_view = dst
    dst_hbm = tile_hbm(dst_view, tuple(dst_view.shape), iter_order=RowMajor())

    Store(dst=dst_hbm, src=src_ts).execute()
