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

"""TensorCopy primitive: SBUF to SBUF copy between TileStreams."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class TensorCopy(nl.NKIObject):
    """Copy data between TileStreams in SBUF."""

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
    ) -> None:
        self._dst = dst
        self._src = src
        self._name = f"TensorCopy(dst={dst.get_name()}, src={src.get_name()})"

        kernel_assert(
            dst.get_num_tiles() == src.get_num_tiles(),
            f"TensorCopy '{self._name}': tile count mismatch - dst={dst.get_num_tiles()}, src={src.get_num_tiles()}",
        )

    def execute(self) -> None:
        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()

        for _ in range(self._dst.get_num_tiles()):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()
            nisa.tensor_copy(dst=dst_tile, src=src_tile)

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()


def tensor_copy(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
) -> None:
    """Compact tensor_copy: SBUF to SBUF. Whole tensor, no tiling.

    Args:
        dst: Destination tensor in SBUF
        src: Source tensor in SBUF
    """
    logical_shape = get_logical_shape(dst)
    dst_ts = tile_stream.tile(dst, logical_shape, iter_order=RowMajor())
    src_ts = tile_stream.tile(src, logical_shape, iter_order=RowMajor())
    TensorCopy(dst=dst_ts, src=src_ts).execute()
