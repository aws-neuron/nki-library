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
NKI Primitives QuantizeMX Module.

Provides MX (microscaling) quantization using nisa.quantize_mx.
Converts bf16/fp32 tensors to mxfp8_x4 or mxfp4_x4 packed format with block-wise scales.

MX Quantization:
    - Block-wise: every 32 elements share one scale
    - Output is packed: 4 values per element (_x4 suffix)
    - Scale is uint8

Usage:
    dst = TileStream((P, F), (P, F), nl.float8_e4m3fn_x4, name="quantized")
    dst_scale = TileStream((P, F), (P, F), nl.uint8, name="scale")
    blas.QuantizeMX(dst=dst, dst_scale=dst_scale, src=src).execute()
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.logging import get_logger
from ..tile_stream import TileStream


class QuantizeMX(nl.NKIObject):
    """MX quantization: quantize src (bf16/fp32) to dst (mxfp8_x4/mxfp4_x4) + dst_scale (uint8).

    Source F dimension must be 4x destination F dimension (due to x4 packing).
    Scale shape must match dst shape exactly.
    """

    def __init__(
        self,
        dst: TileStream,
        dst_scale: TileStream,
        src: TileStream,
    ) -> None:
        self._dst = dst
        self._dst_scale = dst_scale
        self._src = src
        self._name = f"QuantizeMX(src={src.get_name()}, dst={dst.get_name()})"
        self._logger = get_logger(self._name)

        # Validate shapes
        src_shape = self._src.get_logical_shape()
        dst_shape = self._dst.get_logical_shape()
        scale_shape = self._dst_scale.get_logical_shape()
        src_tile = self._src.get_tile_shape()
        dst_tile = self._dst.get_tile_shape()

        kernel_assert(
            len(src_shape) == len(dst_shape),
            f"QuantizeMX '{self._name}' expects same number of dims, got src={len(src_shape)}, dst={len(dst_shape)}",
        )

        # All dimensions except last (F) must match
        kernel_assert(
            src_shape[:-1] == dst_shape[:-1],
            f"QuantizeMX '{self._name}' batch/P dimensions must match, got src={src_shape[:-1]}, dst={dst_shape[:-1]}",
        )

        # Source F must be 4x dest F (due to x4 packing)
        kernel_assert(
            src_shape[-1] == dst_shape[-1] * 4,
            f"QuantizeMX '{self._name}' src F must be 4x dst F (x4 packing), got src[-1]={src_shape[-1]}, dst[-1]={dst_shape[-1]}",
        )

        # Scale shape must match dst shape
        kernel_assert(
            scale_shape == dst_shape,
            f"QuantizeMX '{self._name}' scale shape must match dst shape, got scale={scale_shape}, dst={dst_shape}",
        )

        # Validate tile shapes: src tile F must be 4x dst tile F
        kernel_assert(
            src_tile[:-1] == dst_tile[:-1],
            f"QuantizeMX '{self._name}' tile batch/P dimensions must match, got src_tile={src_tile[:-1]}, dst_tile={dst_tile[:-1]}",
        )
        kernel_assert(
            src_tile[-1] == dst_tile[-1] * 4,
            f"QuantizeMX '{self._name}' src tile F must be 4x dst tile F (x4 packing), got src_tile[-1]={src_tile[-1]}, dst_tile[-1]={dst_tile[-1]}",
        )

        # Store number of tiles for iteration
        self._num_tiles = self._src.get_num_tiles()
        kernel_assert(
            self._num_tiles == self._dst.get_num_tiles(),
            f"QuantizeMX '{self._name}' src and dst must have same number of tiles",
        )

        self._logger.debug(f"src_shape={src_shape}, dst_shape={dst_shape}, num_tiles={self._num_tiles}")

    def execute(self) -> None:
        self._src.reset_cur_tile()
        self._dst.reset_cur_tile()
        self._dst_scale.reset_cur_tile()

        for _ in range(self._num_tiles):
            src_tile = self._src.get_tile()
            dst_tile = self._dst.get_tile()
            scale_tile = self._dst_scale.get_tile()

            nisa.quantize_mx(
                dst=dst_tile,
                src=src_tile,
                dst_scale=scale_tile,
            )

        self._src.reset_cur_tile()
        self._dst.reset_cur_tile()
        self._dst_scale.reset_cur_tile()

    def get_name(self) -> str:
        return self._name
