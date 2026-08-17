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

"""Down projection sub-kernel for MLP TKG with LHS/RHS swap mode."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import MLPParameters
from .mlp_tkg_constants import (
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsDownTileCounts,
)
from .mlp_tkg_utils import adaptive_dge_mode


def down_projection_lhs_rhs_swap(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    output_tile: nl.NkiTensor,
    weight_tiles: list[nl.NkiTensor],
    bias_tile: nl.NkiTensor,
    dequant_tile: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
):
    """
    Performs a single Down projection shard on the H using regular matmult with operands swapped.

    All weight/bias inputs are pre-sharded NkiTensor instances — callers handle LNC/shard slicing.

    Computes: Weight[I, H_per_shard] @ Hidden[I, T] + Optional(bias_tile) → [T, H_per_shard]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Args:
        hidden (nl.NkiTensor): [I0, I1, T] — hidden activations in SBUF
        weight (NkiTensor): [I, H_per_shard] — pre-sharded weight matrix
        output_tile (nl.NkiTensor): [H0, H1_shard, T] — output buffer in SBUF

    Returns:
        Output tensor with shape [128, H//128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I = weight.shape[0]
    H = dims.H_per_shard
    H0 = dims.H0
    num_Itiles = div_ceil(I, I0)

    weight_base_idx = tiles.weight_base_idx

    # ---------- Compute matmul ----------
    perBankT = dims._psum_fmax // T
    num_required_down_psum_banks = div_ceil(dims.H1_shard, perBankT)
    kernel_assert(
        num_required_down_psum_banks <= dims._psum_bmax,
        f"Required psum banks for down projection: {num_required_down_psum_banks}, which exceeds hardware limit of {dims._psum_bmax}, please use CTE mode",
    )

    # Allocate PSUM buffers
    result_psums = []
    for psum_idx in range(num_required_down_psum_banks):
        result_psum = nl.ndarray(
            (dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"down_psum_{sbm.get_name_prefix()}_{psum_idx}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    for hidden_tiles in TiledRange(H, tiles.HTile):
        h1_offset = hidden_tiles.start_offset // H0

        weight_hbm_tile_slice = weight.slice(dim=1, start=hidden_tiles.start_offset, end=hidden_tiles.end_offset)

        for i_tile in TiledRange(I, I0):
            # Load weight [I0, HTile] from HBM into SBUF tile
            weight_idx = (weight_base_idx + hidden_tiles.index * num_Itiles + i_tile.index) % tiles.num_allocated_w_tile

            hidden_sb_tile_slice = hidden.slice(dim=0, start=0, end=i_tile.size).slice(
                dim=1, start=i_tile.index, end=i_tile.index + 1
            )
            weight_sb_tile_slice = (
                weight_tiles[weight_idx]
                .slice(dim=0, start=0, end=i_tile.size)
                .slice(dim=1, start=0, end=hidden_tiles.size)
            )

            nisa.dma_copy(
                dst=weight_sb_tile_slice,
                src=weight_hbm_tile_slice.slice(dim=0, start=i_tile.start_offset, end=i_tile.end_offset),
                dge_mode=adaptive_dge_mode(weight),
            )

            """
            When use_tkg_down_proj_optimized_layout is disabled, the weight-tile access pattern
            uses a stride of H1_shard. When enabled, the framework permutes the weights so
            that the weight tile can be accessed without any stride.
            """
            if params.use_tkg_down_proj_optimized_layout:
                for h1_tile in TiledRange(hidden_tiles.size, H0):
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=weight_sb_tile_slice.slice(
                            dim=1, start=h1_tile.start_offset, end=h1_tile.end_offset
                        ),
                        moving=hidden_sb_tile_slice,
                    )
            else:
                num_Htiles = div_ceil(hidden_tiles.size, H0)
                weight_reshaped = weight_sb_tile_slice.reshape_dim(dim=1, shape=(H0, num_Htiles)).permute(
                    dims=(0, 2, 1)
                )
                for h1_tile in TiledRange(hidden_tiles.size, H0):
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=weight_reshaped.select(dim=1, index=h1_tile.index),
                        moving=hidden_sb_tile_slice,
                    )

    # Reshape output to 2D for PSUM copy (skip flatten if already 2D)
    if output_tile.ndim > 2:
        output_tile_nd = output_tile.flatten_dims(start_dim=1, end_dim=2)
    else:
        output_tile_nd = output_tile

    # Copy PSUM output to SBUF
    perBankElem = perBankT * T
    for psum_tiles in TiledRange(dims.H1_shard, perBankT):
        numElements = psum_tiles.size * T

        if params.quant_params.is_quant_row():
            dequant_tile_view = (
                dequant_tile.slice(
                    dim=1, start=psum_tiles.index * psum_tiles.size, end=(psum_tiles.index + 1) * psum_tiles.size
                )
                .expand_dim(dim=2)
                .broadcast(dim=2, size=T)
            )
        else:
            dequant_tile_view = dequant_tile

        interleave_copy(
            index=psum_tiles.index,
            dst=output_tile_nd.slice(
                dim=1, start=psum_tiles.index * perBankElem, end=psum_tiles.index * perBankElem + numElements
            ),
            src=result_psums[psum_tiles.index][0:H0, 0:numElements],
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias ----------
    is_bias = bias_tile is not None
    if is_bias:
        bias_tile_broadcasted = bias_tile.expand_dim(dim=2).broadcast(dim=2, size=T)
        output_tile_2d = output_tile_nd
        nisa.tensor_tensor(
            dst=output_tile_2d,
            data1=output_tile_2d,
            data2=bias_tile_broadcasted,
            op=nl.add,
        )
