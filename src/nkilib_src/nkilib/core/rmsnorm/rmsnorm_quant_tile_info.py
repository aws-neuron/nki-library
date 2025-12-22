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

"""Tile information structures for the RMS Norm Quantization kernel."""

from dataclasses import dataclass

import nki.language as nl

from ..utils.tile_info import TiledDimInfo


@dataclass(frozen=True)
class RMSNormQuantTileInfo(nl.NKIObject):
    """
    Tile information for the RMS Norm Quantization kernel.

    Args:
        outer_dim_tile (TiledDimInfo): Tile info for outer dimensions
        proc_dim_tile (TiledDimInfo): Tile info for processing dimension
    """

    outer_dim_tile: TiledDimInfo
    proc_dim_tile: TiledDimInfo


def build_rms_norm_quant_tile_info(
    processing_shape: tuple[int, int],
) -> RMSNormQuantTileInfo:
    """
    Factory method to construct RMSNormQuantTileInfo.

    Tiles the outer dimension into partition-sized tiles. The processing dimension
    is tiled when applying RMS norm for PE broadcasting of gamma.

    Args:
        processing_shape (tuple[int, int]): (outer_dim_size, proc_dim_size)

    Returns:
        RMSNormQuantTileInfo: Initialized tile info for the kernel
    """
    outer_dim_tile = TiledDimInfo.build(processing_shape[0], nl.tile_size.pmax)
    proc_dim_tile = TiledDimInfo.build(processing_shape[1], nl.tile_size.gemm_moving_fmax)

    return RMSNormQuantTileInfo(outer_dim_tile, proc_dim_tile)
