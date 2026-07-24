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
from ....core.utils.tensor_view import TensorView
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class Activation(nl.NKIObject):
    """
    Element-wise activation primitive using nisa.activation.

    Supports operations like copy (cast), silu, exp, sin, rsqrt, etc.

    Scale and bias are handled symmetrically; each can be:
    - None: no term (scale defaults to 1.0; bias omitted)
    - float: a scalar applied to all elements
    - nl.ndarray / TensorView: a per-partition (P, 1) vector; normalized internally
      into a TileStream so it resolves to the 2D (P, 1) view nisa.activation needs
    - TileStream: a (P, 1) per-partition vector

    A (P, 1) vector scale/bias is broadcast (P, 1) -> (P, F) by nisa.activation itself
    (each partition's single value is applied across that partition's whole row). Vector
    scale/bias tiles along the partition loop in lockstep with dst/src, so P > pmax
    (multi-tile) is supported.

    One hardware asymmetry (enforced by nisa.activation): a vector `scale` must be float32,
    whereas a vector `bias` may be any supported dtype.
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
        op=nl.copy,
        scale: Optional[Union[float, nl.ndarray, TensorView, TileStream]] = None,
        bias: Optional[Union[float, nl.ndarray, TensorView, TileStream]] = None,
    ) -> None:
        self._name = f"Activation(dst={dst.get_name()}, src={src.get_name()}, op={op})"
        self._dst = dst
        self._src = src
        self._op = op
        self._num_tiles = dst.get_num_tiles()

        # scale and bias are handled symmetrically: each may be None, a scalar, a (P, 1)
        # vector tensor/TensorView, or a TileStream. Normalize + validate both the same way.
        self._scale = self._normalize_term("scale", scale)
        self._bias = self._normalize_term("bias", bias)

    def _normalize_term(self, name, term):
        """Normalize a scale/bias term to None / scalar / single-per-tile-shape TileStream,
        and validate a vector term is a (P, 1) per-partition vector matching dst.

        A raw tensor / TensorView from alloc_logical is a 3D container (pdim, n_p_tiles, F);
        nisa.activation wants a 2D per-tile (P, 1) view, so wrap it into a TileStream (whose
        get_tile() resolves that view). A vector term must be (P, 1) with partition size and
        tile count matching dst so it advances in lockstep across the tile loop -- validate
        here rather than let a mis-shaped tensor produce a cryptic downstream nisa error."""
        if term is None or isinstance(term, (int, float)):
            return term
        if not isinstance(term, TileStream):
            term = tile_stream.tile(term, get_logical_shape(term), iter_order=RowMajor())

        dst_p = self._dst.get_tile_shape()[0]
        t_shape = term.get_tile_shape()
        kernel_assert(
            len(t_shape) == 2 and t_shape[0] == dst_p and t_shape[1] == 1,
            f"Activation '{self._name}': {name} must be a (P, 1) vector with P == dst "
            f"partition {dst_p}, got tile shape {t_shape}",
        )
        kernel_assert(
            term.get_num_tiles() == self._num_tiles,
            f"Activation '{self._name}': {name} tile count {term.get_num_tiles()} must "
            f"match dst tile count {self._num_tiles}",
        )
        return term

    def _term_value(self, term, default):
        """Resolve a normalized term for the current tile: `default` (None -> scale 1.0 /
        no bias) for None, the scalar itself, else the current tile's (P, 1) view. Called
        inside the loop so a (P, 1) vector advances with the partition loop."""
        if term is None:
            return default
        if isinstance(term, (int, float)):
            return term
        return term.get_tile().get_view()  # TileStream

    def execute(self) -> None:
        for stream in (self._dst, self._src, self._scale, self._bias):
            if isinstance(stream, TileStream):
                stream.reset_cur_tile()

        for _ in range(self._num_tiles):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()

            nisa.activation(
                dst=dst_tile.get_view(),
                op=self._op,
                data=src_tile.get_view(),
                scale=self._term_value(self._scale, 1.0),
                bias=self._term_value(self._bias, None),
            )

        for stream in (self._dst, self._src, self._scale, self._bias):
            if isinstance(stream, TileStream):
                stream.reset_cur_tile()


def activation(
    dst: Union[TensorView, nl.ndarray],
    src: Union[TensorView, nl.ndarray] = None,
    op=nl.copy,
    scale: Optional[Union[float, TensorView, nl.ndarray]] = None,
    bias: Optional[Union[float, TensorView, nl.ndarray]] = None,
) -> None:
    """Compact activation: dst = op(src * scale + bias). Whole tensor, no tiling.

    scale and bias may each be a scalar, a (P, 1) vector tensor, or None. If src is None,
    operates in-place (src = dst).
    """
    if src is None:
        src = dst
    logical_shape = get_logical_shape(dst)
    dst_ts = tile_stream.tile(dst, logical_shape, iter_order=RowMajor())
    src_ts = tile_stream.tile(src, logical_shape, iter_order=RowMajor())
    # scale and bias (scalar or (P, 1) tensor) are normalized by Activation.
    Activation(dst=dst_ts, src=src_ts, op=op, scale=scale, bias=bias).execute()
