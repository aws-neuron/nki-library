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

"""Reciprocal primitive for nkiprimitives."""

import nki.isa as nisa
import nki.language as nl

from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class Reciprocal(nl.NKIObject):
    """Element-wise reciprocal: dst = 1 / src.

    Supports in-place operation when dst == src.
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
    ) -> None:
        self._dst = dst
        self._src = src
        self._num_tiles = dst.get_num_tiles()

    def execute(self) -> None:
        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()

        for _ in range(self._num_tiles):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()

            nisa.reciprocal(
                dst=dst_tile,
                data=src_tile,
            )

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()


def reciprocal(dst: nl.NkiTensor, src: nl.NkiTensor = None) -> None:
    """Compact reciprocal: dst = 1 / src. Whole tensor, no tiling.

    If src is None, operates in-place (src = dst).
    """
    if src is None:
        src = dst
    dst_ts = tile_stream.tile(dst, get_logical_shape(dst), iter_order=RowMajor())
    src_ts = tile_stream.tile(src, get_logical_shape(src), iter_order=RowMajor())
    Reciprocal(dst=dst_ts, src=src_ts).execute()
