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


from dataclasses import dataclass

import nki.language as nl
from nki.language import NKIObject
from ..utils.tile_info import TiledDimInfo


#
#
# Primary tuple that holds all tile information needed for the kernel
#
@dataclass(frozen=True)
class RMSNormQuantTileInfo(NKIObject):
    # Tile information for the outer dimensions
    outer_dim_tile: TiledDimInfo
    # Tile information for the processed (normalized/quantized) dimension
    proc_dim_tile: TiledDimInfo

    # Factory methods
    # ONLY CONSTRUCT THIS USING THE FACTORY METHODS BELOW


def build_rms_norm_quant_tile_info(
    processing_shape: tuple[int, int],
) -> RMSNormQuantTileInfo:
    # We tile the outer dim into partition dimension sized tiles.  We don't do any subtiling
    # because that actually hurts performance.  We can keep the resources busy simply by
    # working on a tile at a time of size [pmax, PD] where PD is the processing dimension.
    outer_dim_tile = TiledDimInfo.build(processing_shape[0], nl.tile_size.pmax)
    # We tile the processing dimension when applying RMS norm.  Specifically, when
    # multiplying by the gamma because we use the PE to broadcast the gamma vector across
    # the partition dimension.  By tiling in this dimension plus using the PE, we can do some
    # computation overlap with the PE and other engines.  Since we are using the PE, the
    # tile size is limited to the max moving free dimension size.
    proc_dim_tile = TiledDimInfo.build(processing_shape[1], nl.tile_size.gemm_moving_fmax)

    return RMSNormQuantTileInfo(outer_dim_tile, proc_dim_tile)
