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
NKI Primitives TensorTensor Module

Provides element-wise binary operations between two tensors.
"""

import nki.isa as nisa
import nki.language as nl

from ..tile_stream import TileStream


class TensorTensor(nl.NKIObject):
    """
    Element-wise binary operation primitive using nisa.tensor_tensor.

    Computes: dst = op(src1, src2)

    Supported operations:
        - nl.multiply: dst = src1 * src2
        - nl.add: dst = src1 + src2
        - nl.subtract: dst = src1 - src2
        - nl.maximum: dst = max(src1, src2)
        - nl.minimum: dst = min(src1, src2)
    """

    def __init__(
        self,
        dst: TileStream,
        src1: TileStream,
        src2: TileStream,
        op=nl.multiply,
    ) -> None:
        self._name = f"TensorTensor(dst={dst.get_name()}, src1={src1.get_name()}, src2={src2.get_name()}, op={op})"
        self._dst = dst
        self._src1 = src1
        self._src2 = src2
        self._op = op

    def execute_tile(self):
        dst_tile = self._dst.get_tile()
        src1_tile = self._src1.get_tile()
        src2_tile = self._src2.get_tile()

        nisa.tensor_tensor(
            dst=dst_tile.get_view(),
            op=self._op,
            data1=src1_tile.get_view(),
            data2=src2_tile.get_view(),
        )

    def execute(self):
        for _ in range(self._dst.get_num_tiles()):
            self.execute_tile()
        self._dst.reset_cur_tile()
        self._src1.reset_cur_tile()
        self._src2.reset_cur_tile()
