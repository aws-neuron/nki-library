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

"""TensorScalar primitive for nkiprimitives."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import TileStream, get_logical_shape


class TensorScalar(nl.NKIObject):
    """Element-wise tensor-scalar operations.

    Applies one or two scalar operations to each element:
        dst = op1(op0(src, operand0), operand1)

    Common uses:
    - Clipping: op0=nl.minimum, operand0=max_val, op1=nl.maximum, operand1=min_val
    - Scalar multiply: op0=nl.multiply, operand0=scalar
    - Scalar add: op0=nl.add, operand0=scalar
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
        op0,
        operand0: float,
        op1=None,
        operand1: Optional[float] = None,
    ) -> None:
        self._dst = dst
        self._src = src
        self._op0 = op0
        self._operand0 = operand0
        self._op1 = op1
        self._operand1 = operand1
        self._num_tiles = dst.get_num_tiles()

    def execute(self) -> None:
        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()

        for _ in range(self._num_tiles):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()

            if self._op1 is not None:
                nisa.tensor_scalar(
                    dst=dst_tile,
                    data=src_tile,
                    op0=self._op0,
                    operand0=self._operand0,
                    op1=self._op1,
                    operand1=self._operand1,
                )
            else:
                nisa.tensor_scalar(
                    dst=dst_tile,
                    data=src_tile,
                    op0=self._op0,
                    operand0=self._operand0,
                )

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()


def tensor_scalar(
    dst: nl.NkiTensor,
    src: nl.NkiTensor = None,
    op0=None,
    operand0: float = 0.0,
    op1=None,
    operand1: Optional[float] = None,
) -> None:
    """Compact tensor_scalar: dst = op1(op0(src, operand0), operand1). Whole tensor, no tiling.

    If src is None, operates in-place (src = dst).
    """
    if src is None:
        src = dst
    logical_shape = get_logical_shape(dst)
    dst_ts = tile_stream.tile(dst, logical_shape, iter_order=RowMajor())
    src_ts = tile_stream.tile(src, logical_shape, iter_order=RowMajor())
    TensorScalar(dst=dst_ts, src=src_ts, op0=op0, operand0=operand0, op1=op1, operand1=operand1).execute()
