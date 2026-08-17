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
NKI Primitives Activation Module

Provides element-wise activation operations and convenience wrappers.
"""

from typing import Optional, Union

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class Activation(nl.NKIObject):
    """
    Element-wise activation primitive using nisa.activation.

    Supports operations like copy (cast), silu, exp, sin, rsqrt, etc.

    Scale can be:
    - None: no scaling (defaults to 1.0)
    - float: scalar scale applied to all elements
    - nl.NkiTensor: raw tensor, used directly with nisa.activation
    - nl.NkiTensor: tensor view, get_view() called for nisa.activation
    - TileStream: must be single tile, broadcasts (P, 1) -> (P, F)
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
        op=nl.copy,
        scale: Optional[Union[float, nl.NkiTensor, nl.NkiTensor, TileStream]] = None,
        bias: Optional[TileStream] = None,
    ) -> None:
        self._name = f"Activation(dst={dst.get_name()}, src={src.get_name()}, op={op})"
        self._dst = dst
        self._src = src
        self._op = op
        self._scale = scale
        self._bias = bias
        self._num_tiles = dst.get_num_tiles()

        if isinstance(scale, TileStream):
            kernel_assert(
                scale.get_num_tiles() == 1,
                f"Activation scale TileStream must be single tile, got {scale.get_num_tiles()}",
            )

    def _get_scale_value(self):
        """Resolve scale to a value usable by nisa.activation."""
        if self._scale is None:
            return 1.0
        elif isinstance(self._scale, (int, float)):
            return self._scale
        elif isinstance(self._scale, TileStream):
            self._scale.reset_cur_tile()
            return self._scale.get_tile()
        elif isinstance(self._scale, nl.NkiTensor):
            return self._scale
        else:
            # Assume raw nl.NkiTensor
            return self._scale

    def execute(self) -> None:
        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()
        if self._bias is not None:
            self._bias.reset_cur_tile()

        scale_val = self._get_scale_value()

        for _ in range(self._num_tiles):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()
            bias_tile = self._bias.get_tile() if self._bias is not None else None

            nisa.activation(
                dst=dst_tile,
                op=self._op,
                data=src_tile,
                scale=scale_val,
                bias=bias_tile if bias_tile is not None else None,
            )

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()
        if self._bias is not None:
            self._bias.reset_cur_tile()
        if isinstance(self._scale, TileStream):
            self._scale.reset_cur_tile()


def activation(
    dst: nl.NkiTensor,
    src: nl.NkiTensor = None,
    op=nl.copy,
    scale: Optional[Union[float, nl.NkiTensor, nl.NkiTensor]] = None,
    bias: Optional[nl.NkiTensor] = None,
) -> None:
    """Compact activation: dst = op(src * scale + bias). Whole tensor, no tiling.

    If src is None, operates in-place (src = dst).
    """
    if src is None:
        src = dst
    logical_shape = get_logical_shape(dst)
    dst_ts = tile_stream.tile(dst, logical_shape, iter_order=RowMajor())
    src_ts = tile_stream.tile(src, logical_shape, iter_order=RowMajor())
    # Scale as raw tensor/float, bias as TileStream if provided
    bias_ts = tile_stream.tile(bias, get_logical_shape(bias), iter_order=RowMajor()) if bias is not None else None
    Activation(dst=dst_ts, src=src_ts, op=op, scale=scale, bias=bias_ts).execute()
